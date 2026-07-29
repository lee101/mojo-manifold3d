"""Covered manifold3d-compatible mesh and solid API."""

from __future__ import annotations

from enum import Enum
import subprocess
from typing import Callable, Sequence
import warnings

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

from ._lib import addr, f64, i64, lib

_CSG_RESOLUTION = 40


class OpType(Enum):
    Add = 0
    Subtract = 1
    Intersect = 2


class Error(Enum):
    NoError = 0
    NonFiniteVertex = 1
    NotManifold = 2
    VertexOutOfBounds = 3
    PropertiesWrongLength = 4
    MissingPositionProperties = 5
    MergeVectorsDifferentLengths = 6
    MergeIndexOutOfBounds = 7
    TransformWrongLength = 8
    RunIndexWrongLength = 9
    FaceIDWrongLength = 10
    InvalidConstruction = 11
    ResultTooLarge = 12
    InvalidTangents = 13
    Cancelled = 14


def set_csg_resolution(resolution: int) -> None:
    """Set grid cells across the longest CSG result axis (minimum 12)."""
    global _CSG_RESOLUTION
    resolution = int(resolution)
    if resolution < 12:
        raise ValueError("resolution must be at least 12")
    _CSG_RESOLUTION = resolution


class Mesh:
    def __init__(
        self,
        vert_properties,
        tri_verts,
        merge_from_vert=None,
        merge_to_vert=None,
        run_index=None,
        run_original_id=None,
        run_transform=None,
        run_flags=None,
        face_id=None,
        halfedge_tangent=None,
        tolerance: float = 0,
    ):
        vertices = np.asarray(vert_properties)
        faces = np.asarray(tri_verts)
        if vertices.ndim != 2:
            raise ValueError("vert_properties must be a two-dimensional array")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("tri_verts must have shape (n, 3)")
        if np.issubdtype(faces.dtype, np.integer) and faces.size:
            limits = np.iinfo(np.int32)
            if faces.min() < limits.min or faces.max() > limits.max:
                raise OverflowError("tri_verts values must fit in int32")
        elif not np.issubdtype(faces.dtype, np.integer):
            raise TypeError("tri_verts must contain integers")
        if np.issubdtype(vertices.dtype, np.complexfloating):
            raise TypeError("vert_properties must contain real numbers")
        if np.issubdtype(vertices.dtype, np.number) and vertices.size:
            limit = np.finfo(np.float32).max
            finite = np.isfinite(vertices)
            if np.any(finite & (np.abs(vertices) > limit)):
                raise OverflowError("vert_properties values must fit in float32")
        elif not np.issubdtype(vertices.dtype, np.number):
            raise TypeError("vert_properties must contain numbers")
        self._vert_properties = np.ascontiguousarray(vertices, dtype=np.float32)
        self._tri_verts = np.ascontiguousarray(faces, dtype=np.int32)
        self._merge_from_vert = (
            [] if merge_from_vert is None else list(merge_from_vert)
        )
        self._merge_to_vert = [] if merge_to_vert is None else list(merge_to_vert)
        self._run_index = [] if run_index is None else list(run_index)
        self._run_original_id = (
            [] if run_original_id is None else list(run_original_id)
        )
        self._run_transform = np.ascontiguousarray(
            run_transform if run_transform is not None else np.empty((0, 4, 3)),
            dtype=np.float32,
        )
        self._run_flags = [] if run_flags is None else list(run_flags)
        self._face_id = [] if face_id is None else list(face_id)
        self._halfedge_tangent = np.ascontiguousarray(
            halfedge_tangent
            if halfedge_tangent is not None
            else np.empty((0, 3, 4)),
            dtype=np.float32,
        )
        self.tolerance = float(tolerance)

    @property
    def vert_properties(self):
        return self._vert_properties

    @property
    def tri_verts(self):
        return self._tri_verts

    @property
    def merge_from_vert(self):
        return self._merge_from_vert

    @property
    def merge_to_vert(self):
        return self._merge_to_vert

    @property
    def run_index(self):
        return self._run_index

    @property
    def run_original_id(self):
        return self._run_original_id

    @property
    def run_transform(self):
        return self._run_transform

    @property
    def run_flags(self):
        return self._run_flags

    @property
    def face_id(self):
        return self._face_id

    @property
    def halfedge_tangent(self):
        return self._halfedge_tangent

    def merge(self) -> bool:
        return False

    def has_normals(self, run: int) -> bool:
        return False

    def backside(self, run: int) -> bool:
        return False


def _edge_status(vertices: np.ndarray, faces: np.ndarray) -> Error:
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        return Error.MissingPositionProperties
    if not np.isfinite(vertices[:, :3]).all():
        return Error.NonFiniteVertex
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        return Error.VertexOutOfBounds
    if not len(faces):
        return Error.NoError
    if np.any(
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    ):
        return Error.NotManifold
    first = np.concatenate((faces[:, 0], faces[:, 1], faces[:, 2]))
    second = np.concatenate((faces[:, 1], faces[:, 2], faces[:, 0]))
    low = np.minimum(first, second)
    high = np.maximum(first, second)
    edge_keys = low * np.int64(len(vertices)) + high
    unique, inverse, counts = np.unique(
        edge_keys, return_inverse=True, return_counts=True
    )
    if len(unique) == 0 or np.any(counts != 2):
        return Error.NotManifold
    signs = np.where(first < second, 1, -1)
    if np.any(np.bincount(inverse, weights=signs, minlength=len(unique)) != 0):
        return Error.NotManifold
    return Error.NoError


def _compact(vertices: np.ndarray, faces: np.ndarray):
    if not len(faces):
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.int64),
        )
    used, inverse = np.unique(faces, return_inverse=True)
    return np.ascontiguousarray(vertices[used]), np.ascontiguousarray(
        inverse.reshape((-1, 3)), dtype=np.int64
    )


def _oriented(vertices: np.ndarray, faces: np.ndarray):
    vertices, faces = _compact(vertices, faces)
    if not len(faces):
        return vertices, faces
    signed = np.einsum(
        "ij,ij->i",
        vertices[faces[:, 0]],
        np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]]),
    ).sum()
    if signed < 0:
        faces = faces[:, [0, 2, 1]].copy()
    return vertices, faces


def _convex_hull(points) -> "Manifold":
    points = f64(points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        return Manifold()
    try:
        hull = ConvexHull(points)
    except QhullError:
        return Manifold()
    faces = np.ascontiguousarray(hull.simplices, dtype=np.int64)
    a = points[faces[:, 0]]
    b = points[faces[:, 1]]
    c = points[faces[:, 2]]
    reverse = np.einsum(
        "ij,ij->i", np.cross(b - a, c - a), hull.equations[:, :3]
    ) < 0
    faces[reverse] = faces[reverse][:, [0, 2, 1]]
    used = np.asarray(hull.vertices)
    remap = np.empty(len(points), dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    vertices = np.ascontiguousarray(points[used])
    faces = np.ascontiguousarray(remap[faces])
    return Manifold._from_trusted(vertices, faces, convex=True)


class Manifold:
    def __init__(self, mesh: Mesh | None = None):
        self._status = Error.NoError
        self._vertices = np.empty((0, 3), dtype=np.float64)
        self._faces = np.empty((0, 3), dtype=np.int64)
        self._convex_hint: bool | None = True
        if mesh is None:
            return
        if not isinstance(mesh, Mesh):
            raise TypeError("Manifold expects a Mesh")
        vertices = f64(mesh.vert_properties[:, :3], copy=True)
        faces = i64(mesh.tri_verts, copy=True)
        self._status = _edge_status(vertices, faces)
        if self._status is Error.NoError:
            self._vertices, self._faces = _oriented(vertices, faces)
            self._convex_hint = None

    @classmethod
    def _from_trusted(cls, vertices, faces, *, convex: bool | None = None):
        result = cls()
        result._vertices = f64(vertices).reshape((-1, 3))
        result._faces = i64(faces).reshape((-1, 3))
        result._convex_hint = convex
        return result

    @classmethod
    def _from_arrays(cls, vertices, faces, *, validate: bool = True):
        result = cls()
        vertices = f64(vertices, copy=True).reshape((-1, 3))
        faces = i64(faces, copy=True).reshape((-1, 3))
        vertices, faces = _oriented(vertices, faces)
        status = _edge_status(vertices, faces) if validate else Error.NoError
        result._status = status
        if status is Error.NoError:
            result._vertices = vertices
            result._faces = faces
            result._convex_hint = None
        return result

    @staticmethod
    def cube(size=(1.0, 1.0, 1.0), center: bool = False):
        size = np.asarray(size, dtype=np.float64)
        if size.shape != (3,):
            raise ValueError("size must have three components")
        if np.any(size <= 0):
            return Manifold()
        vertices = np.array(
            [
                [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
            ],
            dtype=np.float64,
        ) * size
        if center:
            vertices -= size / 2
        faces = np.array(
            [
                [0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6],
                [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7],
                [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5],
            ],
            dtype=np.int64,
        )
        return Manifold._from_trusted(vertices, faces, convex=True)

    @staticmethod
    def tetrahedron():
        vertices = np.array(
            [[-1, -1, 1], [-1, 1, -1], [1, -1, -1], [1, 1, 1]],
            dtype=np.float64,
        )
        faces = np.array(
            [[2, 0, 1], [0, 3, 1], [2, 3, 0], [3, 2, 1]], dtype=np.int64
        )
        return Manifold._from_trusted(vertices, faces, convex=True)

    @staticmethod
    def cylinder(
        height: float,
        radius_low: float,
        radius_high: float = -1.0,
        circular_segments: int = 0,
        center: bool = False,
    ):
        if radius_high < 0:
            radius_high = radius_low
        segments = int(circular_segments) if circular_segments else 32
        if height <= 0 or radius_low < 0 or radius_high < 0 or segments < 3:
            return Manifold()
        if radius_low == 0 and radius_high == 0:
            return Manifold()
        angles = np.arange(segments) * (2 * np.pi / segments)
        ring = np.column_stack((np.cos(angles), np.sin(angles)))
        z0 = -height / 2 if center else 0.0
        vertices = []
        bottom = []
        top = []
        if radius_low > 0:
            bottom = list(range(len(vertices), len(vertices) + segments))
            vertices.extend(
                np.column_stack(
                    (ring * radius_low, np.full(segments, z0))
                ).tolist()
            )
        else:
            bottom = [len(vertices)]
            vertices.append([0.0, 0.0, z0])
        if radius_high > 0:
            top = list(range(len(vertices), len(vertices) + segments))
            vertices.extend(
                np.column_stack(
                    (ring * radius_high, np.full(segments, z0 + height))
                ).tolist()
            )
        else:
            top = [len(vertices)]
            vertices.append([0.0, 0.0, z0 + height])
        faces = []
        if len(bottom) == segments:
            for i in range(1, segments - 1):
                faces.append([bottom[0], bottom[i + 1], bottom[i]])
        if len(top) == segments:
            for i in range(1, segments - 1):
                faces.append([top[0], top[i], top[i + 1]])
        for i in range(segments):
            j = (i + 1) % segments
            if len(bottom) == 1:
                faces.append([bottom[0], top[j], top[i]])
            elif len(top) == 1:
                faces.append([bottom[i], bottom[j], top[0]])
            else:
                faces.extend(
                    ([bottom[i], bottom[j], top[j]], [bottom[i], top[j], top[i]])
                )
        return Manifold._from_trusted(vertices, faces, convex=True)

    @staticmethod
    def sphere(radius: float, circular_segments: int = 0):
        if radius <= 0:
            return Manifold()
        segments = max(4, int(circular_segments) if circular_segments else 24)
        segments = ((segments + 3) // 4) * 4
        frequency = segments // 4
        base = np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
            dtype=np.float64,
        )
        octants = [
            (4, 0, 2), (4, 2, 1), (4, 1, 3), (4, 3, 0),
            (5, 2, 0), (5, 1, 2), (5, 3, 1), (5, 0, 3),
        ]
        points = []
        for a, b, c in octants:
            for i in range(frequency + 1):
                for j in range(frequency + 1 - i):
                    k = frequency - i - j
                    point = (i * base[a] + j * base[b] + k * base[c]) / frequency
                    points.append(point / np.linalg.norm(point) * radius)
        rounded = np.round(np.asarray(points), 14)
        points = np.unique(rounded, axis=0)
        return _convex_hull(points)

    @staticmethod
    def hull_points(pts):
        return _convex_hull(pts)

    @staticmethod
    def batch_hull(manifolds: Sequence["Manifold"]):
        points = [solid._vertices for solid in manifolds if not solid.is_empty()]
        return _convex_hull(np.vstack(points)) if points else Manifold()

    @staticmethod
    def batch_boolean(manifolds: Sequence["Manifold"], op: OpType):
        solids = list(manifolds)
        if not solids:
            return Manifold()
        result = solids[0]
        for solid in solids[1:]:
            if op is OpType.Add:
                result = result + solid
            elif op is OpType.Subtract:
                result = result - solid
            elif op is OpType.Intersect:
                result = result ^ solid
            else:
                raise ValueError("unknown operation")
        return result

    @staticmethod
    def compose(manifolds: Sequence["Manifold"]):
        return Manifold.batch_boolean(manifolds, OpType.Add)

    def status(self):
        return self._status

    def is_empty(self) -> bool:
        return len(self._faces) == 0

    def to_mesh(self, normal_idx: int = -1):
        return Mesh(
            np.ascontiguousarray(self._vertices, dtype=np.float32),
            np.ascontiguousarray(self._faces, dtype=np.uint32),
        )

    def num_vert(self) -> int:
        return len(self._vertices)

    def num_prop_vert(self) -> int:
        return len(self._vertices)

    def num_tri(self) -> int:
        return len(self._faces)

    def num_edge(self) -> int:
        if self.is_empty():
            return 0
        edges = np.concatenate(
            (
                self._faces[:, [0, 1]],
                self._faces[:, [1, 2]],
                self._faces[:, [2, 0]],
            )
        )
        return len(np.unique(np.sort(edges, axis=1), axis=0))

    def num_prop(self) -> int:
        return 0

    def bounding_box(self):
        if self.is_empty():
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        low = self._vertices.min(axis=0)
        high = self._vertices.max(axis=0)
        return tuple(np.concatenate((low, high)).tolist())

    def _measure(self):
        if self.is_empty():
            return np.zeros(2)
        result = np.zeros(2, dtype=np.float64)
        lib().mm_measure(
            addr(self._vertices), addr(self._faces), len(self._faces),
            addr(result, writable=True)
        )
        return result

    def volume(self) -> float:
        return float(self._measure()[0])

    def surface_area(self) -> float:
        return float(self._measure()[1])

    def genus(self) -> int:
        if self.is_empty():
            return 0
        parent = np.arange(len(self._vertices))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b, c in self._faces:
            root = find(int(a))
            for item in (int(b), int(c)):
                other = find(item)
                if root != other:
                    parent[other] = root
        components = len({find(i) for i in range(len(parent))})
        chi = self.num_vert() - self.num_edge() + self.num_tri()
        return max(0, int(round((2 * components - chi) / 2)))

    def _apply_matrix(self, matrix):
        if self.is_empty():
            return Manifold()
        matrix = f64(matrix).reshape((3, 4))
        vertices = np.empty_like(self._vertices)
        lib().mm_transform(
            addr(self._vertices), addr(vertices, writable=True), addr(matrix),
            len(vertices)
        )
        faces = self._faces
        if np.linalg.det(matrix[:, :3]) < 0:
            faces = faces[:, [0, 2, 1]].copy()
        return Manifold._from_trusted(
            vertices, faces, convex=self._convex_hint
        )

    def transform(self, m):
        return self._apply_matrix(m)

    def translate(self, t):
        t = np.asarray(t, dtype=np.float64)
        if t.shape != (3,):
            raise ValueError("translation must have three components")
        matrix = np.column_stack((np.eye(3), t))
        return self._apply_matrix(matrix)

    def scale(self, v):
        values = np.asarray(v, dtype=np.float64)
        if values.ndim == 0:
            values = np.repeat(values, 3)
        if values.shape != (3,):
            raise ValueError("scale must be a scalar or three components")
        matrix = np.column_stack((np.diag(values), np.zeros(3)))
        return self._apply_matrix(matrix)

    def rotate(self, v):
        angles = np.radians(np.asarray(v, dtype=np.float64))
        if angles.shape != (3,):
            raise ValueError("rotation must have three components")
        sx, sy, sz = np.sin(angles)
        cx, cy, cz = np.cos(angles)
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return self._apply_matrix(np.column_stack((rz @ ry @ rx, np.zeros(3))))

    def mirror(self, v):
        normal = np.asarray(v, dtype=np.float64)
        length = np.linalg.norm(normal)
        if normal.shape != (3,) or length == 0:
            return Manifold()
        normal /= length
        linear = np.eye(3) - 2 * np.outer(normal, normal)
        return self._apply_matrix(np.column_stack((linear, np.zeros(3))))

    def warp(self, warp_func: Callable):
        vertices = np.array([warp_func(tuple(v)) for v in self._vertices])
        return Manifold._from_arrays(vertices, self._faces)

    def warp_batch(self, warp_func: Callable):
        return Manifold._from_arrays(warp_func(self._vertices.copy()), self._faces)

    def hull(self):
        return _convex_hull(self._vertices)

    def _is_convex(self) -> bool:
        if self._convex_hint is not None:
            return self._convex_hint
        hull = self.hull()
        self._convex_hint = abs(hull.volume() - self.volume()) <= max(
            1.0e-9, hull.volume() * 1.0e-8
        )
        return self._convex_hint

    def minkowski_sum(self, other: "Manifold"):
        if self.is_empty() or other.is_empty():
            return Manifold()
        if not self._is_convex() or not other._is_convex():
            raise NotImplementedError("minkowski_sum currently requires convex inputs")
        points = np.empty(
            (len(self._vertices) * len(other._vertices), 3), dtype=np.float64
        )
        lib().mm_pairwise_sum(
            addr(self._vertices),
            len(self._vertices),
            addr(other._vertices),
            len(other._vertices),
            addr(points, writable=True),
            0,
        )
        return _convex_hull(points)

    def minkowski_difference(self, other: "Manifold"):
        if self.is_empty() or other.is_empty():
            return Manifold()
        if not self._is_convex() or not other._is_convex():
            raise NotImplementedError(
                "minkowski_difference currently requires convex inputs"
            )
        hull = ConvexHull(self._vertices)
        normals = hull.equations[:, :3]
        offsets = hull.equations[:, 3] + np.max(
            -normals @ other._vertices.T, axis=1
        )
        objective = np.array([0.0, 0.0, 0.0, -1.0])
        constraints = np.column_stack((normals, np.ones(len(normals))))
        solution = linprog(
            objective,
            A_ub=constraints,
            b_ub=-offsets,
            bounds=[(None, None)] * 4,
            method="highs",
        )
        if not solution.success or solution.x[3] <= 1.0e-10:
            return Manifold()
        halfspaces = np.column_stack((normals, offsets))
        try:
            vertices = HalfspaceIntersection(
                halfspaces, solution.x[:3]
            ).intersections
        except QhullError:
            return Manifold()
        return _convex_hull(vertices)

    def __add__(self, other):
        return _boolean(self, other, OpType.Add)

    def __sub__(self, other):
        return _boolean(self, other, OpType.Subtract)

    def __xor__(self, other):
        return _boolean(self, other, OpType.Intersect)

    def boolean(self, other: "Manifold", op: OpType, *, device: str = "cpu"):
        if not isinstance(op, OpType):
            raise TypeError("op must be an OpType")
        if device not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'")
        return _boolean(self, other, op, device=device)


def _gpu_memory_available(required_bytes: int) -> bool:
    if required_bytes >= 2_000_000_000:
        return False
    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        free_mib = min(
            int(line.strip()) for line in process.stdout.splitlines() if line.strip()
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    return process.returncode == 0 and free_mib >= 4000


def _field(solid: Manifold, origin, spacing, shape):
    nx, ny, nz = shape
    result = np.empty(nx * ny * nz, dtype=np.float64)
    lib().mm_signed_distance(
        addr(solid._vertices),
        addr(solid._faces),
        len(solid._faces),
        float(origin[0]),
        float(origin[1]),
        float(origin[2]),
        float(spacing),
        nx,
        ny,
        nz,
        addr(result, writable=True),
    )
    return result


def _field_pair(first, second, origin, spacing, shape, *, device):
    nx, ny, nz = shape
    count = nx * ny * nz
    first_result = np.empty(count, dtype=np.float64)
    second_result = np.empty(count, dtype=np.float64)
    required_bytes = (
        first._vertices.nbytes
        + first._faces.nbytes
        + second._vertices.nbytes
        + second._faces.nbytes
        + first_result.nbytes
        + second_result.nbytes
    )
    if device == "gpu" and _gpu_memory_available(required_bytes):
        if lib().mm_signed_distance_pair_gpu(
            addr(first._vertices),
            addr(first._faces),
            len(first._vertices),
            len(first._faces),
            addr(second._vertices),
            addr(second._faces),
            len(second._vertices),
            len(second._faces),
            float(origin[0]),
            float(origin[1]),
            float(origin[2]),
            float(spacing),
            nx,
            ny,
            nz,
            addr(first_result, writable=True),
            addr(second_result, writable=True),
        ):
            return first_result, second_result
        warnings.warn(
            "GPU execution failed; using the CPU implementation",
            RuntimeWarning,
            stacklevel=2,
        )
    elif device == "gpu":
        warnings.warn(
            "GPU unavailable or below the memory safety threshold; "
            "using the CPU implementation",
            RuntimeWarning,
            stacklevel=2,
        )
    return (
        _field(first, origin, spacing, shape),
        _field(second, origin, spacing, shape),
    )


def _boolean(
    first: Manifold,
    second: Manifold,
    operation: OpType,
    *,
    device: str = "cpu",
):
    if not isinstance(second, Manifold):
        return NotImplemented
    if first.status() is not Error.NoError:
        return first
    if second.status() is not Error.NoError:
        return second
    if operation is OpType.Add:
        if first.is_empty():
            return second
        if second.is_empty():
            return first
        low = np.minimum(first._vertices.min(axis=0), second._vertices.min(axis=0))
        high = np.maximum(first._vertices.max(axis=0), second._vertices.max(axis=0))
    elif operation is OpType.Subtract:
        if first.is_empty() or second.is_empty():
            return first
        low = first._vertices.min(axis=0)
        high = first._vertices.max(axis=0)
    else:
        if first.is_empty() or second.is_empty():
            return Manifold()
        low = np.maximum(first._vertices.min(axis=0), second._vertices.min(axis=0))
        high = np.minimum(first._vertices.max(axis=0), second._vertices.max(axis=0))
        if np.any(high <= low):
            return Manifold()
    span = high - low
    longest = float(span.max())
    if longest <= 0:
        return Manifold()
    spacing = longest / _CSG_RESOLUTION
    origin = low - 2 * spacing
    upper = high + 2 * spacing
    dimensions = np.ceil((upper - origin) / spacing).astype(int) + 1
    nx, ny, nz = map(int, dimensions)
    first_field, second_field = _field_pair(
        first,
        second,
        origin,
        spacing,
        (nx, ny, nz),
        device=device,
    )
    combined = np.empty_like(first_field)
    lib().mm_combine_fields(
        addr(first_field),
        addr(second_field),
        addr(combined, writable=True),
        len(combined),
        operation.value,
    )
    triangle_count = int(lib().mm_march_count(addr(combined), nx, ny, nz))
    if triangle_count == 0:
        return Manifold()
    triangles = np.empty((triangle_count, 3, 3), dtype=np.float64)
    edge_keys = np.empty((triangle_count, 3), dtype=np.int64)
    emitted = lib().mm_march_emit(
        addr(combined),
        addr(triangles, writable=True),
        addr(edge_keys, writable=True),
        nx,
        ny,
        nz,
        float(origin[0]),
        float(origin[1]),
        float(origin[2]),
        float(spacing),
    )
    if emitted != triangle_count:
        raise RuntimeError("marching-tetrahedra count and emit passes disagreed")
    flat = triangles.reshape((-1, 3))
    _, first_indices, inverse = np.unique(
        edge_keys, return_index=True, return_inverse=True
    )
    vertices = flat[first_indices]
    faces = inverse.reshape((-1, 3))
    keep = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 2] != faces[:, 0])
    )
    faces = np.ascontiguousarray(faces[keep], dtype=np.int64)
    status = _edge_status(vertices, faces)
    if status is not Error.NoError:
        raise RuntimeError("marching tetrahedra produced a non-manifold mesh")
    return Manifold._from_trusted(vertices, faces, convex=None)
