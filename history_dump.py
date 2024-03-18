 1/1: import numpy as np
 1/2:
a = np.asarrray([3,-1,2])
b = np.asarray([4,2,-6])
c = np.asarray([6,9,3])
d = 2*a + 0.5*b - 2/3*c
d
 1/3:
a = np.asarray([3,-1,2])
b = np.asarray([4,2,-6])
c = np.asarray([6,9,3])
d = 2*a + 0.5*b - 2/3*c
d
 1/4:
import vedo
vedo.settings.default_backend= 'vtk'
 1/5: vedo.Point3D?
 1/6: vedo.Points3D?
 1/7: vedo.shapes.Points3D
 1/8: vedo.shapes?
 1/9: vedo.shapes.Points?
1/10: vedo.Points?
1/11: np.stack([d,e,f], axis=0)
1/12:
d = np.asarray([-9, 4, -2])
e = np.asarray([0,1,17])
f = np.asarray([3,4,-3])

de = e-d
df = f-d
ef = f-e

#points = vedo.Points(np.stack([
1/13: np.stack([d,e,f], axis=0)
1/14:
d = np.asarray([-9, 4, -2])
e = np.asarray([0,1,17])
f = np.asarray([3,4,-3])

de = e-d
df = f-d
ef = f-e

points = vedo.Points(np.stack([d,e,f], axis=0), r=4, c='red')

meshes = [points]
plt = vedo.Plotter()
#plt.
1/15: plt.show?
1/16: plt.show(meshes, axes=True)
1/17:
d = np.asarray([-9, 4, -2])
e = np.asarray([0,1,17])
f = np.asarray([3,4,-3])

de = e-d
df = f-d
ef = f-e

points = vedo.Points(np.stack([d,e,f], axis=0), r=4, c='red')

meshes = [points]
plt = vedo.Plotter()
plt.show(meshes, axes=True)
1/18:
d = np.asarray([-9, 4, -2])
e = np.asarray([0,1,17])
f = np.asarray([3,4,-3])

de = e-d
df = f-d
ef = f-e

points = vedo.Points(np.stack([d,e,f], axis=0), r=4, c='red')
lines = vedo.Line(
    np.stack([d,e,f],lw=2)
)

meshes = [points, lines]
plt = vedo.Plotter()
plt.show(meshes, axes=True)
1/19:
d = np.asarray([-9, 4, -2])
e = np.asarray([0,1,17])
f = np.asarray([3,4,-3])

de = e-d
df = f-d
ef = f-e

points = vedo.Points(np.stack([d,e,f], axis=0), r=4, c='red')
lines = vedo.Line(
    np.stack([d,e,f],axis=0), lw=2
)

meshes = [points, lines]
plt = vedo.Plotter()
plt.show(meshes, axes=True)
1/20:
d = np.asarray([-9, 4, -2])
e = np.asarray([0,1,17])
f = np.asarray([3,4,-3])

de = e-d
df = f-d
ef = f-e

points = vedo.Points(np.stack([d,e,f], axis=0), r=4, c='red')
lines = vedo.Line(
    np.stack([d,e,f],axis=0), lw=2, closed=True
)

meshes = [points, lines]
plt = vedo.Plotter()
plt.show(meshes, axes=True)
1/21:
d = np.asarray([-9, 4, -2])
e = np.asarray([0,1,17])
f = np.asarray([3,4,-3])

de = e-d
df = f-d
ef = f-e

points = vedo.Points(np.stack([d,e,f], axis=0), r=4, c='red')
lines = vedo.Line(
    np.stack([d,e,f],axis=0), lw=2, closed=True
)

middle = e + 0.5*ef
m = vedo.Points(middle, r=4, c='blue')

meshes = [points, lines, m]
plt = vedo.Plotter()
plt.show(meshes, axes=True)
1/22: middle
1/23:
d = np.asarray([-9, 4, -2])
e = np.asarray([0,1,17])
f = np.asarray([3,4,-3])

de = e-d
df = f-d
ef = f-e

points = vedo.Points(np.stack([d,e,f], axis=0), r=4, c='red')
lines = vedo.Line(
    np.stack([d,e,f],axis=0), lw=2, closed=True
)

middle = e + 0.5*ef
m = vedo.Points(middle.reshape(-1, 3), r=4, c='blue')

meshes = [points, lines, m]
plt = vedo.Plotter()
plt.show(meshes, axes=True)
1/24:
d = np.asarray([-9, 4, -2])
e = np.asarray([0,1,17])
f = np.asarray([3,4,-3])

de = e-d
df = f-d
ef = f-e

points = vedo.Points(np.stack([d,e,f], axis=0), r=4, c='red')
lines = vedo.Line(
    np.stack([d,e,f],axis=0), lw=2, closed=True
)

middle = e + 0.5*ef
m = vedo.Points(middle.reshape(-1, 3), r=10, c='blue')

meshes = [points, lines, m]
plt = vedo.Plotter()
plt.show(meshes, axes=True)
1/25:
d = np.asarray([-9, 4, -2])
e = np.asarray([0,1,17])
f = np.asarray([3,4,-3])

de = e-d
df = f-d
ef = f-e

points = vedo.Points(np.stack([d,e,f], axis=0), r=4, c='red')
lines = vedo.Line(
    np.stack([d,e,f],axis=0), lw=2, closed=True
)

middle = e + 0.5*ef
m = vedo.Points(middle.reshape(-1, 3), r=10, c='blue')

center = 1/3*(d+e+f)
c = vedo.Points(c.reshape(-1, 3), r=10, c='green7')

meshes = [points, lines, m, c]
plt = vedo.Plotter()
plt.show(meshes, axes=True)
1/26: center
1/27:
d = np.asarray([-9, 4, -2])
e = np.asarray([0,1,17])
f = np.asarray([3,4,-3])

de = e-d
df = f-d
ef = f-e

points = vedo.Points(np.stack([d,e,f], axis=0), r=4, c='red')
lines = vedo.Line(
    np.stack([d,e,f],axis=0), lw=2, closed=True
)

middle = e + 0.5*ef
m = vedo.Points(middle.reshape(-1, 3), r=10, c='blue')

center = 1/3*(d+e+f)
c = vedo.Points(c.reshape(-1, 3), r=10, c='green7')

meshes = [points, lines, m, c]
plt = vedo.Plotter()
plt.show(meshes, axes=True)
1/28:
d = np.asarray([-9, 4, -2])
e = np.asarray([0,1,17])
f = np.asarray([3,4,-3])

de = e-d
df = f-d
ef = f-e

points = vedo.Points(np.stack([d,e,f], axis=0), r=4, c='red')
lines = vedo.Line(
    np.stack([d,e,f],axis=0), lw=2, closed=True
)

middle = e + 0.5*ef
m = vedo.Points(middle.reshape(-1, 3), r=10, c='blue')

center = 1/3*(d+e+f)
c = vedo.Points(center.reshape(-1, 3), r=10, c='green7')

meshes = [points, lines, m, c]
plt = vedo.Plotter()
plt.show(meshes, axes=True)
1/29:
d = np.asarray([-9, 4, -2])
e = np.asarray([0,1,17])
f = np.asarray([3,4,-3])

de = e-d
df = f-d
ef = f-e

points = vedo.Points(np.stack([d,e,f], axis=0), r=4, c='red')
lines = vedo.Line(
    np.stack([d,e,f],axis=0), lw=2, closed=True
)

middle = e + 0.5*ef
m = vedo.Points(middle.reshape(-1, 3), r=10, c='blue')

center = 1/3*(d+e+f)
c = vedo.Points(center.reshape(-1, 3), r=10, c='green7')

line = vedo.Line(middle, center, c='orange5', lw=3)

meshes = [points, lines, m, c, line]
plt = vedo.Plotter()
plt.show(meshes, axes=True)
1/30: vedo.Points?
1/31:
d = np.asarray([-9, 4, -2])
e = np.asarray([0,1,17])
f = np.asarray([3,4,-3])

de = e-d
df = f-d
ef = f-e

points = vedo.Points(np.stack([d,e,f], axis=0), r=4, c='red')
lines = vedo.Line(
    np.stack([d,e,f],axis=0), lw=2, closed=True
)

middle = e + 0.5*ef
m = vedo.Points(middle.reshape(-1, 3), r=10, c='blue')

center = 1/3*(d+e+f)
c = vedo.Points(center.reshape(-1, 3), r=10, c='green7')

line = vedo.Line(middle, center, c='orange5', lw=3)

meshes = [points, lines, m, c, line]
plt = vedo.Plotter()
plt.show(meshes, axes=True)
1/32: center - middle
1/33: middle - center
   1: import numpy as np
   2: np.linalg.norm?
   3: q = np.asarray([0,1,0,0])
   4: v = q / np.linalg.norm(q)
   5: v
   6: np.linalg.norm(q)
   7: q = np.asarray([0,1,1,0])
   8: v = q / np.linalg.norm(q)
   9: v
  10: np.linalg.norm(q)
  11: np.linalg.norm(v)
  12: sqrt(1**2 + 1**2)
  13: np.sqrt(1**2 + 1**2)
  14:
def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)
  15: q
  16:
def normalize_quat(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)
  17: v = normalize_quat(q)
  18: v
  19:
a = np.ndarray([[5],[6],[7]])
b = np.ndarray([[1],[2],[3]])
a*b
  20:
a = np.asarray([[5],[6],[7]])
b = np.asarray([[1],[2],[3]])
a*b
  21: a = np.asarray([[1,0,0,0],[1,1,0,0],[1,1,1,0],[1,1,1,1]])
  22: np.linalg.norm(a)
  23: np.linalg.norm(a, dim=0)
  24: np.linalg.norm(a, axis=0)
  25: np.linalg.norm(a, axis=1)
  26: q = np.asarray([0,1,1,0])
  27:
def normalize_quat(q: np.ndarray) -> np.ndarray:
    if q.shape == 1:
        q = q.reshape(1, -1)
    return q / np.linalg.norm(q, axis=1)
  28:
v = normalize_quat(q)
v
  29:
def normalize_quat(q: np.ndarray) -> np.ndarray:
    if q.ndim == 1:
        q = q.reshape(1, -1)
    return q / np.linalg.norm(q, axis=1)
  30:
v = normalize_quat(q)
v
  31: q
  32:
def normalize_quat(q: np.ndarray) -> np.ndarray:
    single_dim = False
    if q.ndim == 1:
        single_dim = True
        q = q.reshape(1, -1)
    q /= np.linalg.norm(q, axis=1)
    if single_dim:
        return q[0]
    else:
        return q
  33:
v = normalize_quat(q)
v
  34: q = np.asarray([0,1,1,0], dtype=torch.float32)
  35: q = np.asarray([0,1,1,0], dtype=np.float32)
  36:
def normalize_quat(q: np.ndarray) -> np.ndarray:
    single_dim = False
    if q.ndim == 1:
        single_dim = True
        q = q.reshape(1, -1)
    q /= np.linalg.norm(q, axis=1)
    if single_dim:
        return q[0]
    else:
        return q
  37:
v = normalize_quat(q)
v
  38:
def normalize_quat(q: np.ndarray) -> np.ndarray:
    single_dim = False
    if q.ndim == 1:
        single_dim = True
        q = q.reshape(1, -1)
    q /= np.linalg.norm(q, axis=1)
    if single_dim:
        return q[0]
    else:
        return q
  39:
v = normalize_quat(q)
v
  40: q = np.asarray([0,1,1,0])
  41:
def normalize_quat(q: np.ndarray) -> np.ndarray:
    single_dim = False
    if q.ndim == 1:
        single_dim = True
        q = q.reshape(1, -1)
    q /= np.linalg.norm(q, axis=1)
    if single_dim:
        return q[0]
    else:
        return q
  42:
v = normalize_quat(q)
v
  43:
def normalize_quat(q: np.ndarray) -> np.ndarray:
    single_dim = False
    if q.ndim == 1:
        single_dim = True
        q = q.reshape(1, -1)
    q_norm /= np.linalg.norm(q, axis=1)
    if single_dim:
        return q_norm[0]
    else:
        return q_norm
  44:
v = normalize_quat(q)
v
  45:
def normalize_quat(q: np.ndarray) -> np.ndarray:
    single_dim = False
    if q.ndim == 1:
        single_dim = True
        q = q.reshape(1, -1)
    q_norm = q / np.linalg.norm(q, axis=1)
    if single_dim:
        return q_norm[0]
    else:
        return q_norm
  46:
v = normalize_quat(q)
v
  47: q.dtype
  48: q.dtype == np.int32
  49:
v = normalize_quat(q)
v
  50: v.dtype
  51: q = np.asarray([0,1,1,0], [1,0,0,0])
  52: q = np.asarray([[0,1,1,0], [1,0,0,0]])
  53:
def normalize_quat(q: np.ndarray) -> np.ndarray:
    single_dim = False
    if q.ndim == 1:
        single_dim = True
        q = q.reshape(1, -1)
    q_norm = q / np.linalg.norm(q, axis=1)
    if single_dim:
        return q_norm[0]
    else:
        return q_norm
  54:
v = normalize_quat(q)
v
  55:
def normalize_quat(q: np.ndarray) -> np.ndarray:
    single_dim = False
    if q.ndim == 1:
        single_dim = True
        q = q.reshape(1, -1)
    q /= np.linalg.norm(q, axis=1)
    if single_dim:
        return q[0]
    else:
        return q
  56:
v = normalize_quat(q)
v
  57: q = np.asarray([[0,1,1,0], [1,0,0,0]], dtype=np.float32)
  58:
def normalize_quat(q: np.ndarray) -> np.ndarray:
    single_dim = False
    if q.ndim == 1:
        single_dim = True
        q = q.reshape(1, -1)
    q /= np.linalg.norm(q, axis=1)
    if single_dim:
        return q[0]
    else:
        return q
  59:
v = normalize_quat(q)
v
  60:
def normalize_quat(q: np.ndarray) -> np.ndarray:
    single_dim = False
    if q.ndim == 1:
        single_dim = True
        q = q.reshape(1, -1)
    q /= np.linalg.norm(q, axis=1).reshape(-1, 1)
    if single_dim:
        return q[0]
    else:
        return q
  61:
v = normalize_quat(q)
v
  62: q = np.asarray([[0,1,1,0], [1,0,0,0]])
  63: q.dtype == np.int
  64:
def normalize_quat(q: np.ndarray) -> np.ndarray:
    single_dim = False
    if q.ndim == 1:
        single_dim = True
        q = q.reshape(1, -1)
    q_norm = q / np.linalg.norm(q, axis=1).reshape(-1, 1)
    if single_dim:
        return q_norm[0]
    else:
        return q_norm
  65: q = np.asarray([[0,1,1,0], [1,0,0,0]])
  66:
def normalize_quat(q: np.ndarray) -> np.ndarray:
    single_dim = False
    if q.ndim == 1:
        single_dim = True
        q = q.reshape(1, -1)
    q_norm = q / np.linalg.norm(q, axis=1).reshape(-1, 1)
    if single_dim:
        return q_norm[0]
    else:
        return q_norm
  67:
v = normalize_quat(q)
v
  68:
a = np.asarray([[5],[6],[7]])
b = np.asarray([[1],[2],[3]])
a*b - a*b
  69: a = np.asarray([[1,0,0,0],[1,1,0,0],[1,1,1,0],[1,1,1,1]])
  70: a.shape[1] == 4
  71: a.shape[1] == 4 == a.shape[1]
  72: a.shape[1] == 4 == a.shape[0]
  73: np.tile?
  74: q
  75: q[0]
  76: np.tile(q[0], (1, 2))
  77: np.tile(q[0], 1)
  78: np.tile(q[0], 2)
  79: np.tile(q[0], (1,2))
  80: np.tile(q[0], (2,1))
  81:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0])"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0]
    qb = q[:, 1]
    qc = q[:, 2]
    qd = q[:, 3]
    pa = p[:, 0]
    pb = p[:, 1]
    pc = p[:, 2]
    pd = p[:, 3]
    print(qa)
    print(qa * pa)
  82:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0]
    qb = q[:, 1]
    qc = q[:, 2]
    qd = q[:, 3]
    pa = p[:, 0]
    pb = p[:, 1]
    pc = p[:, 2]
    pd = p[:, 3]
    print(qa)
    print(qa * pa)
  83: hamilton_product(q[0], q[1])
  84: q = np.asarray([[1,1,1,0], [1,0,0,0]])
  85: hamilton_product(q[0], q[1])
  86: hamilton_product(q, q)
  87: hamilton_product(q[0], q[0])
  88: hamilton_product(q, q)
  89:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0]
    qb = q[:, 1]
    qc = q[:, 2]
    qd = q[:, 3]
    pa = p[:, 0]
    pb = p[:, 1]
    pc = p[:, 2]
    pd = p[:, 3]
    print(qa * pa)
  90: hamilton_product(q, q)
  91:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0]
    qb = q[:, 1]
    qc = q[:, 2]
    qd = q[:, 3]
    pa = p[:, 0]
    pb = p[:, 1]
    pc = p[:, 2]
    pd = p[:, 3]
    print((qa * pa).reshape(-1, 1))
  92: hamilton_product(q, q)
  93:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[..., 0]
    qb = q[, 1]
    qc = q[:, 2]
    qd = q[:, 3]
    pa = p[..., 0]
    pb = p[:, 1]
    pc = p[:, 2]
    pd = p[:, 3]
    print(qa * pa)
    #print((qa * pa).reshape(-1, 1))
  94:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[..., 0]
    qb = q[:, 1]
    qc = q[:, 2]
    qd = q[:, 3]
    pa = p[..., 0]
    pb = p[:, 1]
    pc = p[:, 2]
    pd = p[:, 3]
    print(qa * pa)
    #print((qa * pa).reshape(-1, 1))
  95: hamilton_product(q, q)
  96:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0]
    qb = q[:, 1]
    qc = q[:, 2]
    qd = q[:, 3]
    pa = p[:, 0]
    pb = p[:, 1]
    pc = p[:, 2]
    pd = p[:, 3]
    res = np.asarray([qa*pa - qb*pb - qc*pc - qd*pd,
                      qa*pb + qb*pa + qc*pd - qd*pc,
                      qa*pc - qb*pd + qc*pa + qd*pb,
                      qa*pd + qb*pc - qc*pb + qd*pa], dtype=np.float64)
    return res
  97: hamilton_product(q[0], q[0])
  98: hamilton_product(q, q)
  99:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0].reshape(-1, 1)
    qb = q[:, 1].reshape(-1, 1)
    qc = q[:, 2].reshape(-1, 1)
    qd = q[:, 3].reshape(-1, 1)
    pa = p[:, 0].reshape(-1, 1)
    pb = p[:, 1].reshape(-1, 1)
    pc = p[:, 2].reshape(-1, 1)
    pd = p[:, 3].reshape(-1, 1)
    res = np.asarray([qa*pa - qb*pb - qc*pc - qd*pd,
                      qa*pb + qb*pa + qc*pd - qd*pc,
                      qa*pc - qb*pd + qc*pa + qd*pb,
                      qa*pd + qb*pc - qc*pb + qd*pa], dtype=np.float64)
    return res
 100: hamilton_product(q, q)
 101: hamilton_product(q[0], q[0])
 102: hamilton_product(q[0], q[0]).shape
 103: hamilton_product(q[0], q[0])
 104:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0]
    qb = q[:, 1]
    qc = q[:, 2]
    qd = q[:, 3]
    pa = p[:, 0]
    pb = p[:, 1]
    pc = p[:, 2]
    pd = p[:, 3]
    res = np.asarray([qa*pa - qb*pb - qc*pc - qd*pd,
                      qa*pb + qb*pa + qc*pd - qd*pc,
                      qa*pc - qb*pd + qc*pa + qd*pb,
                      qa*pd + qb*pc - qc*pb + qd*pa], dtype=np.float64)
    return res
 105: hamilton_product(q[0], q[0])
 106:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0]
    qb = q[:, 1]
    qc = q[:, 2]
    qd = q[:, 3]
    pa = p[:, 0]
    pb = p[:, 1]
    pc = p[:, 2]
    pd = p[:, 3]
    print(qa.shape)
    res = np.asarray([qa*pa - qb*pb - qc*pc - qd*pd,
                      qa*pb + qb*pa + qc*pd - qd*pc,
                      qa*pc - qb*pd + qc*pa + qd*pb,
                      qa*pd + qb*pc - qc*pb + qd*pa], dtype=np.float64)
    return res
 107: hamilton_product(q[0], q[0])
 108:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0]
    qb = q[:, 1]
    qc = q[:, 2]
    qd = q[:, 3]
    pa = p[:, 0]
    pb = p[:, 1]
    pc = p[:, 2]
    pd = p[:, 3]
    print(qa.shape)
    print(qa)
    res = np.asarray([qa*pa - qb*pb - qc*pc - qd*pd,
                      qa*pb + qb*pa + qc*pd - qd*pc,
                      qa*pc - qb*pd + qc*pa + qd*pb,
                      qa*pd + qb*pc - qc*pb + qd*pa], dtype=np.float64)
    return res
 109: hamilton_product(q[0], q[0])
 110:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0]
    qb = q[:, 1]
    qc = q[:, 2]
    qd = q[:, 3]
    pa = p[:, 0]
    pb = p[:, 1]
    pc = p[:, 2]
    pd = p[:, 3]
    print(qa.shape)
    print(qa)
    print(qa*pa)
    print((qa*pa).shape)
    res = np.asarray([qa*pa - qb*pb - qc*pc - qd*pd,
                      qa*pb + qb*pa + qc*pd - qd*pc,
                      qa*pc - qb*pd + qc*pa + qd*pb,
                      qa*pd + qb*pc - qc*pb + qd*pa], dtype=np.float64)
    return res
 111: hamilton_product(q[0], q[0])
 112:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0]
    qb = q[:, 1]
    qc = q[:, 2]
    qd = q[:, 3]
    pa = p[:, 0]
    pb = p[:, 1]
    pc = p[:, 2]
    pd = p[:, 3]
    print(qa.shape)
    print(qa)
    print(qa*pa)
    print((qa*pa).shape)
    res = np.asarray([qa*pa - qb*pb - qc*pc - qd*pd,
                      qa*pb + qb*pa + qc*pd - qd*pc,
                      qa*pc - qb*pd + qc*pa + qd*pb,
                      qa*pd + qb*pc - qc*pb + qd*pa], dtype=np.float64)
    print(res.shape)
    return res
 113: hamilton_product(q[0], q[0])
 114:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0]
    qb = q[:, 1]
    qc = q[:, 2]
    qd = q[:, 3]
    pa = p[:, 0]
    pb = p[:, 1]
    pc = p[:, 2]
    pd = p[:, 3]
    print(qa.shape)
    print(qa)
    print(qa*pa)
    print((qa*pa).shape)
    res = np.asarray([qa*pa - qb*pb - qc*pc - qd*pd, qa*pb + qb*pa + qc*pd - qd*pc, qa*pc - qb*pd + qc*pa + qd*pb, qa*pd + qb*pc - qc*pb + qd*pa], dtype=np.float64)
    print(res.shape)
    return res
 115: hamilton_product(q[0], q[0])
 116:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0]
    qb = q[:, 1]
    qc = q[:, 2]
    qd = q[:, 3]
    pa = p[:, 0]
    pb = p[:, 1]
    pc = p[:, 2]
    pd = p[:, 3]
    print(qa.shape)
    print(qa)
    print(qa*pa)
    print((qa*pa).shape)
    res = np.asarray(
        [
            qa*pa - qb*pb - qc*pc - qd*pd,
            qa*pb + qb*pa + qc*pd - qd*pc,
            qa*pc - qb*pd + qc*pa + qd*pb,
            qa*pd + qb*pc - qc*pb + qd*pa
        ], dtype=np.float64).reshape(-1, 4)
    print(res.shape)
    return res
 117: hamilton_product(q[0], q[0])
 118: hamilton_product(q, q[0])
 119:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0]
    qb = q[:, 1]
    qc = q[:, 2]
    qd = q[:, 3]
    pa = p[:, 0]
    pb = p[:, 1]
    pc = p[:, 2]
    pd = p[:, 3]
    res = np.asarray(
        [
            qa*pa - qb*pb - qc*pc - qd*pd,
            qa*pb + qb*pa + qc*pd - qd*pc,
            qa*pc - qb*pd + qc*pa + qd*pb,
            qa*pd + qb*pc - qc*pb + qd*pa
        ], dtype=np.float64).reshape(-1, 4)
    return res
 120: hamilton_product(q, q[0])
 121: q
 122: hamilton_product(q[0], q[0])
 123: hamilton_product(q, q[0])
 124:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    print(p)
    qa = q[:, 0]
    qb = q[:, 1]
    qc = q[:, 2]
    qd = q[:, 3]
    pa = p[:, 0]
    pb = p[:, 1]
    pc = p[:, 2]
    pd = p[:, 3]
    res = np.asarray(
        [
            qa*pa - qb*pb - qc*pc - qd*pd,
            qa*pb + qb*pa + qc*pd - qd*pc,
            qa*pc - qb*pd + qc*pa + qd*pb,
            qa*pd + qb*pc - qc*pb + qd*pa
        ], dtype=np.float64).reshape(-1, 4)
    return res
 125: hamilton_product(q, q[0])
 126:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    print(p)
    qa = q[:, 0]
    print(qa*pa)
    qb = q[:, 1]
    qc = q[:, 2]
    qd = q[:, 3]
    pa = p[:, 0]
    pb = p[:, 1]
    pc = p[:, 2]
    pd = p[:, 3]
    res = np.asarray(
        [
            qa*pa - qb*pb - qc*pc - qd*pd,
            qa*pb + qb*pa + qc*pd - qd*pc,
            qa*pc - qb*pd + qc*pa + qd*pb,
            qa*pd + qb*pc - qc*pb + qd*pa
        ], dtype=np.float64).reshape(-1, 4)
    return res
 127: hamilton_product(q, q[0])
 128:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    print(p)
    qa = q[:, 0]
    qb = q[:, 1]
    qc = q[:, 2]
    qd = q[:, 3]
    pa = p[:, 0]
    pb = p[:, 1]
    pc = p[:, 2]
    pd = p[:, 3]
    print(qa*pa)
    res = np.asarray(
        [
            qa*pa - qb*pb - qc*pc - qd*pd,
            qa*pb + qb*pa + qc*pd - qd*pc,
            qa*pc - qb*pd + qc*pa + qd*pb,
            qa*pd + qb*pc - qc*pb + qd*pa
        ], dtype=np.float64).reshape(-1, 4)
    return res
 129: hamilton_product(q, q[0])
 130:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    print(f"q={q}")
    print(f"p={p}")
    qa = q[:, 0]
    qb = q[:, 1]
    qc = q[:, 2]
    qd = q[:, 3]
    pa = p[:, 0]
    pb = p[:, 1]
    pc = p[:, 2]
    pd = p[:, 3]
    print(qa*pa)
    res = np.asarray(
        [
            qa*pa - qb*pb - qc*pc - qd*pd,
            qa*pb + qb*pa + qc*pd - qd*pc,
            qa*pc - qb*pd + qc*pa + qd*pb,
            qa*pd + qb*pc - qc*pb + qd*pa
        ], dtype=np.float64).reshape(-1, 4)
    return res
 131: hamilton_product(q, q[0])
 132:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    print(f"q={q}")
    print(f"p={p}")
    qa = q[:, 0]
    qb = q[:, 1]
    qc = q[:, 2]
    qd = q[:, 3]
    pa = p[:, 0]
    pb = p[:, 1]
    pc = p[:, 2]
    pd = p[:, 3]
    print(qb*pb)
    res = np.asarray(
        [
            qa*pa - qb*pb - qc*pc - qd*pd,
            qa*pb + qb*pa + qc*pd - qd*pc,
            qa*pc - qb*pd + qc*pa + qd*pb,
            qa*pd + qb*pc - qc*pb + qd*pa
        ], dtype=np.float64).reshape(-1, 4)
    return res
 133: hamilton_product(q, q[0])
 134:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    print(f"q={q}")
    print(f"p={p}")
    qa = q[:, 0].reshape(-1, 1)
    qb = q[:, 1].reshape(-1, 1)
    qc = q[:, 2].reshape(-1, 1)
    qd = q[:, 3].reshape(-1, 1)
    pa = p[:, 0].reshape(-1, 1)
    pb = p[:, 1].reshape(-1, 1)
    pc = p[:, 2].reshape(-1, 1)
    pd = p[:, 3].reshape(-1, 1)
    print(qb*pb)
    res = np.asarray(
        [
            qa*pa - qb*pb - qc*pc - qd*pd,
            qa*pb + qb*pa + qc*pd - qd*pc,
            qa*pc - qb*pd + qc*pa + qd*pb,
            qa*pd + qb*pc - qc*pb + qd*pa
        ], dtype=np.float64).reshape(-1, 4)
    return res
 135: hamilton_product(q, q[0])
 136:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    print(f"q={q}")
    print(f"p={p}")
    qa = q[:, 0].reshape(-1, 1)
    qb = q[:, 1].reshape(-1, 1)
    qc = q[:, 2].reshape(-1, 1)
    qd = q[:, 3].reshape(-1, 1)
    pa = p[:, 0].reshape(-1, 1)
    pb = p[:, 1].reshape(-1, 1)
    pc = p[:, 2].reshape(-1, 1)
    pd = p[:, 3].reshape(-1, 1)
    print(qa*pa - qb*pb)
    res = np.asarray(
        [
            qa*pa - qb*pb - qc*pc - qd*pd,
            qa*pb + qb*pa + qc*pd - qd*pc,
            qa*pc - qb*pd + qc*pa + qd*pb,
            qa*pd + qb*pc - qc*pb + qd*pa
        ], dtype=np.float64).reshape(-1, 4)
    return res
 137: hamilton_product(q, q[0])
 138:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    print(f"q={q}")
    print(f"p={p}")
    qa = q[:, 0].reshape(-1, 1)
    qb = q[:, 1].reshape(-1, 1)
    qc = q[:, 2].reshape(-1, 1)
    qd = q[:, 3].reshape(-1, 1)
    pa = p[:, 0].reshape(-1, 1)
    pb = p[:, 1].reshape(-1, 1)
    pc = p[:, 2].reshape(-1, 1)
    pd = p[:, 3].reshape(-1, 1)
    print(qa*pa - qb*pb)
    res = np.asarray(
        [
            qa*pa - qb*pb - qc*pc - qd*pd,
            qa*pb + qb*pa + qc*pd - qd*pc,
            qa*pc - qb*pd + qc*pa + qd*pb,
            qa*pd + qb*pc - qc*pb + qd*pa
        ], dtype=np.float64)
    return res
 139: hamilton_product(q, q[0])
 140: hamilton_product(q, q[0]).shape
 141:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0].reshape(-1, 1)
    qb = q[:, 1].reshape(-1, 1)
    qc = q[:, 2].reshape(-1, 1)
    qd = q[:, 3].reshape(-1, 1)
    pa = p[:, 0].reshape(-1, 1)
    pb = p[:, 1].reshape(-1, 1)
    pc = p[:, 2].reshape(-1, 1)
    pd = p[:, 3].reshape(-1, 1)
    res = np.asarray(
        [
            qa*pa - qb*pb - qc*pc - qd*pd,
            qa*pb + qb*pa + qc*pd - qd*pc,
            qa*pc - qb*pd + qc*pa + qd*pb,
            qa*pd + qb*pc - qc*pb + qd*pa
        ], dtype=np.float64)
    return res
 142: hamilton_product(q, q[0]).shape
 143:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0].reshape(-1, 1)
    qb = q[:, 1].reshape(-1, 1)
    qc = q[:, 2].reshape(-1, 1)
    qd = q[:, 3].reshape(-1, 1)
    pa = p[:, 0].reshape(-1, 1)
    pb = p[:, 1].reshape(-1, 1)
    pc = p[:, 2].reshape(-1, 1)
    pd = p[:, 3].reshape(-1, 1)
    print((qa*pa).shape)
    res = np.asarray(
        [
            qa*pa - qb*pb - qc*pc - qd*pd,
            qa*pb + qb*pa + qc*pd - qd*pc,
            qa*pc - qb*pd + qc*pa + qd*pb,
            qa*pd + qb*pc - qc*pb + qd*pa
        ], dtype=np.float64)
    return res
 144: hamilton_product(q, q[0]).shape
 145:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0].reshape(-1, 1)
    qb = q[:, 1].reshape(-1, 1)
    qc = q[:, 2].reshape(-1, 1)
    qd = q[:, 3].reshape(-1, 1)
    pa = p[:, 0].reshape(-1, 1)
    pb = p[:, 1].reshape(-1, 1)
    pc = p[:, 2].reshape(-1, 1)
    pd = p[:, 3].reshape(-1, 1)
    print((qa*pa - qb*pb - qc*pc - qd*pd).shape)
    res = np.asarray(
        [
            qa*pa - qb*pb - qc*pc - qd*pd,
            qa*pb + qb*pa + qc*pd - qd*pc,
            qa*pc - qb*pd + qc*pa + qd*pb,
            qa*pd + qb*pc - qc*pb + qd*pa
        ], dtype=np.float64)
    return res
 146: hamilton_product(q, q[0]).shape
 147:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0].reshape(-1, 1)
    qb = q[:, 1].reshape(-1, 1)
    qc = q[:, 2].reshape(-1, 1)
    qd = q[:, 3].reshape(-1, 1)
    pa = p[:, 0].reshape(-1, 1)
    pb = p[:, 1].reshape(-1, 1)
    pc = p[:, 2].reshape(-1, 1)
    pd = p[:, 3].reshape(-1, 1)
    print((qa*pa - qb*pb - qc*pc - qd*pd).shape)
    res = np.concatenate(
        [
            qa*pa - qb*pb - qc*pc - qd*pd,
            qa*pb + qb*pa + qc*pd - qd*pc,
            qa*pc - qb*pd + qc*pa + qd*pb,
            qa*pd + qb*pc - qc*pb + qd*pa
        ], dtype=np.float64)
    return res
 148: hamilton_product(q, q[0]).shape
 149: hamilton_product(q, q[0])
 150:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0].reshape(-1, 1)
    qb = q[:, 1].reshape(-1, 1)
    qc = q[:, 2].reshape(-1, 1)
    qd = q[:, 3].reshape(-1, 1)
    pa = p[:, 0].reshape(-1, 1)
    pb = p[:, 1].reshape(-1, 1)
    pc = p[:, 2].reshape(-1, 1)
    pd = p[:, 3].reshape(-1, 1)
    print((qa*pa - qb*pb - qc*pc - qd*pd).shape)
    res = np.concatenate(
        [
            qa*pa - qb*pb - qc*pc - qd*pd,
            qa*pb + qb*pa + qc*pd - qd*pc,
            qa*pc - qb*pd + qc*pa + qd*pb,
            qa*pd + qb*pc - qc*pb + qd*pa
        ], axis=0, dtype=np.float64)
    return res
 151: hamilton_product(q, q[0])
 152:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0].reshape(-1, 1)
    qb = q[:, 1].reshape(-1, 1)
    qc = q[:, 2].reshape(-1, 1)
    qd = q[:, 3].reshape(-1, 1)
    pa = p[:, 0].reshape(-1, 1)
    pb = p[:, 1].reshape(-1, 1)
    pc = p[:, 2].reshape(-1, 1)
    pd = p[:, 3].reshape(-1, 1)
    print((qa*pa - qb*pb - qc*pc - qd*pd).shape)
    res = np.concatenate(
        [
            qa*pa - qb*pb - qc*pc - qd*pd,
            qa*pb + qb*pa + qc*pd - qd*pc,
            qa*pc - qb*pd + qc*pa + qd*pb,
            qa*pd + qb*pc - qc*pb + qd*pa
        ], axis=1, dtype=np.float64)
    return res
 153: hamilton_product(q, q[0])
 154:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0].reshape(-1, 1)
    qb = q[:, 1].reshape(-1, 1)
    qc = q[:, 2].reshape(-1, 1)
    qd = q[:, 3].reshape(-1, 1)
    pa = p[:, 0].reshape(-1, 1)
    pb = p[:, 1].reshape(-1, 1)
    pc = p[:, 2].reshape(-1, 1)
    pd = p[:, 3].reshape(-1, 1)
    res = np.concatenate(
        [
            qa*pa - qb*pb - qc*pc - qd*pd,
            qa*pb + qb*pa + qc*pd - qd*pc,
            qa*pc - qb*pd + qc*pa + qd*pb,
            qa*pd + qb*pc - qc*pb + qd*pa
        ], axis=1, dtype=np.float64)
    return res
 155: hamilton_product(q, q[0])
 156: hamilton_product(q, q)
 157: q
 158: normalize_quat(q)
 159:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0].reshape(-1, 1)
    qb = q[:, 1].reshape(-1, 1)
    qc = q[:, 2].reshape(-1, 1)
    qd = q[:, 3].reshape(-1, 1)
    pa = p[:, 0].reshape(-1, 1)
    pb = p[:, 1].reshape(-1, 1)
    pc = p[:, 2].reshape(-1, 1)
    pd = p[:, 3].reshape(-1, 1)
    res = np.concatenate(
        [
            qa*pa - qb*pb - qc*pc - qd*pd,
            qa*pb + qb*pa + qc*pd - qd*pc,
            qa*pc - qb*pd + qc*pa + qd*pb,
            qa*pd + qb*pc - qc*pb + qd*pa
        ], axis=1, dtype=np.float64)
    return res
 160: hamilton_product(q, q)
 161:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0].reshape(-1, 1)
    qb = q[:, 1].reshape(-1, 1)
    qc = q[:, 2].reshape(-1, 1)
    qd = q[:, 3].reshape(-1, 1)
    pa = p[:, 0].reshape(-1, 1)
    pb = p[:, 1].reshape(-1, 1)
    pc = p[:, 2].reshape(-1, 1)
    pd = p[:, 3].reshape(-1, 1)
    res = np.concatenate(
        [
            qa*pa - qb*pb - qc*pc - qd*pd,
            qa*pb + qb*pa + qc*pd - qd*pc,
            qa*pc - qb*pd + qc*pa + qd*pb,
            qa*pd + qb*pc - qc*pb + qd*pa
        ], axis=1, dtype=np.float64)
    if q_single and p_single:
        return res[0]
    return res
 162: hamilton_product(q[0], q[0])
 163:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0].reshape(-1, 1)
    qb = q[:, 1].reshape(-1, 1)
    qc = q[:, 2].reshape(-1, 1)
    qd = q[:, 3].reshape(-1, 1)
    pa = p[:, 0].reshape(-1, 1)
    pb = p[:, 1].reshape(-1, 1)
    pc = p[:, 2].reshape(-1, 1)
    pd = p[:, 3].reshape(-1, 1)
    res = np.concatenate(
        [
            qa*pa - qb*pb - qc*pc - qd*pd,
            qa*pb + qb*pa + qc*pd - qd*pc,
            qa*pc - qb*pd + qc*pa + qd*pb,
            qa*pd + qb*pc - qc*pb + qd*pa
        ], axis=1, dtype=np.float64)
    #if q_single and p_single:
    #    return res[0]
    return res
 164: hamilton_product(q[0], q[0])
 165:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.ndim == p.ndim, f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0].reshape(-1, 1)
    qb = q[:, 1].reshape(-1, 1)
    qc = q[:, 2].reshape(-1, 1)
    qd = q[:, 3].reshape(-1, 1)
    pa = p[:, 0].reshape(-1, 1)
    pb = p[:, 1].reshape(-1, 1)
    pc = p[:, 2].reshape(-1, 1)
    pd = p[:, 3].reshape(-1, 1)
    res = np.concatenate(
        [
            qa*pa - qb*pb - qc*pc - qd*pd,
            qa*pb + qb*pa + qc*pd - qd*pc,
            qa*pc - qb*pd + qc*pa + qd*pb,
            qa*pd + qb*pc - qc*pb + qd*pa
        ], axis=1, dtype=np.float64)
    if q_single and p_single:
        return res[0]
    return res
 166: hamilton_product(q[0], q[0])
 167: hamilton_product(q[0], q)
 168: def quat_norm(q: np.ndarray) -> double:
 169:
def quat_norm(q: np.ndarray) -> double:
    pass
 170:
def quat_norm(q: np.ndarray) -> float:
    pass
 171:
def quat_norm(q: np.ndarray) -> float:
    single = False
    if q.ndim == 1:
        single = True
        q = q.reshape(1, -1)
    norm = np.linalg.norm(q, axis=1).reshape(-1, 1)
    if single:
        return norm[0]
    return norm
 172: quat_norm(q)
 173: quat_norm(q[0])
 174: quat_norm(q)
 175:
def normalize_quat(q: np.ndarray) -> np.ndarray:
    """
        Normalizes (possibly batched) quaternion q
    """
    single_dim = False
    if q.ndim == 1:
        single_dim = True
        q = q.reshape(1, -1)
    q_norm = q / quat_norm
    if single_dim:
        return q_norm[0]
    else:
        return q_norm
 176: normalize_quat(q)
 177:
def normalize_quat(q: np.ndarray) -> np.ndarray:
    """
        Normalizes (possibly batched) quaternion q
    """
    single_dim = False
    if q.ndim == 1:
        single_dim = True
        q = q.reshape(1, -1)
    q_norm = q / quat_norm(q)
    if single_dim:
        return q_norm[0]
    else:
        return q_norm
 178: normalize_quat(q)
 179: quat_norm(q)**2
 180:
def conjugate_p_by_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
        Calculates conjugation q * p * q^-1 of (possibly batched) quaternions p and q
        If p is a vector quaternion representing a position P, and q is a unit quaternion
        representing a rotation R, the result is a vector quaternion representing the new
        position when point P is rotated by R
    """
    q_inv = q / quat_norm(q)**2
    print(q_inv)
 181: conjugate_p_by_q(q[0], q[1])
 182:
def conjugate_p_by_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
        Calculates conjugation q * p * q^-1 of (possibly batched) quaternions p and q
        If p is a vector quaternion representing a position P, and q is a unit quaternion
        representing a rotation R, the result is a vector quaternion representing the new
        position when point P is rotated by R
    """
    q_inv = q / quat_norm(q)
    print(q_inv)
 183: conjugate_p_by_q(q[0], q[1])
 184: q
 185: conjugate_p_by_q(q[0], q[0])
 186:
def conjugate_p_by_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
        Calculates conjugation q * p * q^-1 of (possibly batched) quaternions p and q
        If p is a vector quaternion representing a position P, and q is a unit quaternion
        representing a rotation R, the result is a vector quaternion representing the new
        position when point P is rotated by R
    """
    q_inv = q / quat_norm(q)**2
    print(q_inv)
 187: conjugate_p_by_q(q[0], q[0])
 188: conjugate_p_by_q(q[0], q)
 189:
def conjugate_p_by_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
        Calculates conjugation q * p * q^-1 of (possibly batched) quaternions p and q
        If p is a vector quaternion representing a position P, and q is a unit quaternion
        representing a rotation R, the result is a vector quaternion representing the new
        position when point P is rotated by R
    """
    q_inv = q / quat_norm(q)**2
    print(q_inv)
    print(hamilton_product(q, q_inv)
 190:
def conjugate_p_by_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
        Calculates conjugation q * p * q^-1 of (possibly batched) quaternions p and q
        If p is a vector quaternion representing a position P, and q is a unit quaternion
        representing a rotation R, the result is a vector quaternion representing the new
        position when point P is rotated by R
    """
    q_inv = q / quat_norm(q)**2
    print(q_inv)
    print(hamilton_product(q, q_inv))
 191:
def conjugate_p_by_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
        Calculates conjugation q * p * q^-1 of (possibly batched) quaternions p and q
        If p is a vector quaternion representing a position P, and q is a unit quaternion
        representing a rotation R, the result is a vector quaternion representing the new
        position when point P is rotated by R
    """
    q_inv = q / quat_norm(q)**2
    #print(q_inv)
    print(hamilton_product(q, q_inv))
 192: conjugate_p_by_q(q[0], q)
 193:
def conjugate_p_by_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
        Calculates conjugation q * p * q^-1 of (possibly batched) quaternions p and q
        If p is a vector quaternion representing a position P, and q is a unit quaternion
        representing a rotation R, the result is a vector quaternion representing the new
        position when point P is rotated by R
    """
    q_inv = q / quat_norm(q)**2
    print(q_inv)
    print(hamilton_product(q, q_inv))
 194: conjugate_p_by_q(q[0], q)
 195: np.concatenate?
 196: q[:, 0]
 197: -q[:, 0]
 198:
def conjugate_quat(q: np.ndarray) -> np.ndarray:
    """
        Returns the conjugate of q=(a+bi+cj+dk) as q*=(a-bi-cj-dk)
    """
    single = False
    if q.ndim == 1:
        single = True
        q = q.reshape(1, -1)
    conjugate = np.concatenate((q[:, 0].reshape(-1, 1),
                                -q[:, 1].reshape(-1, 1),
                                -q[:, 2].reshape(-1, 1),
                                -q[:, 3].reshape(-1, 1)
                               ), axis=1, dtype=np.float64)
    if single:
        return conjugate[0]
    return conjugate
 199: conjugate_quat(q)
 200: conjugate_quat(q[0])
 201: conjugate_quat(q)
 202:
def conjugate_p_by_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
        Calculates conjugation q * p * q^-1 of (possibly batched) quaternions p and q
        If p is a vector quaternion representing a position P, and q is a unit quaternion
        representing a rotation R, the result is a vector quaternion representing the new
        position when point P is rotated by R
    """
    q_inv = conjugate_quat(q) / quat_norm(q)**2
    print(q_inv)
    print(hamilton_product(q, q_inv))
 203: conjugate_p_by_q(q[0], q)
 204: conjugate_p_by_q(q[0], q[1])
 205: conjugate_p_by_q(q[0], q[0])
 206:
def conjugate_p_by_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
        Calculates conjugation q * p * q^-1 of (possibly batched) quaternions p and q
        If p is a vector quaternion representing a position P, and q is a unit quaternion
        representing a rotation R, the result is a vector quaternion representing the new
        position when point P is rotated by R
    """
    q_inv = conjugate_quat(q) / quat_norm(q)**2
    print(q_inv)
    print(hamilton_product(q, q_inv))
 207: conjugate_p_by_q(q[0], q[0])
 208: conjugate_p_by_q(q[0], q)
 209: q = np.asarray([[1,1,1,0], [1,0,0,0], [1,0,1,1]])
 210: conjugate_p_by_q(q[0], q)
 211:
def conjugate_p_by_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
        Calculates conjugation q * p * q^-1 of (possibly batched) quaternions p and q
        If p is a vector quaternion representing a position P, and q is a unit quaternion
        representing a rotation R, the result is a vector quaternion representing the new
        position when point P is rotated by R
    """
    q_inv = conjugate_quat(q) / quat_norm(q)**2
    print(q_inv)
    qp = hamilton_product(q, p)
    qpq_inv = hamilton_product(qp, q_inv)
    print(qpq_inv)
    pq_inv = hamilton_product(p, q_inv)
    qpq_inv = hamilton_product(q, pq_inv)
    print(qpq_inv)
 212: conjugate_p_by_q(q[0], q)
 213:
def conjugate_p_by_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
        Calculates conjugation q * p * q^-1 of (possibly batched) quaternions p and q
        If p is a vector quaternion representing a position P, and q is a unit quaternion
        representing a rotation R, the result is a vector quaternion representing the new
        position when point P is rotated by R
    """
    q_inv = conjugate_quat(q) / quat_norm(q)**2
    qp = hamilton_product(q, p)
    qpq_inv = hamilton_product(qp, q_inv)
    print(qpq_inv)
    pq_inv = hamilton_product(p, q_inv)
    qpq_inv = hamilton_product(q, pq_inv)
    print(qpq_inv)
 214: conjugate_p_by_q(q[0], q)
 215:
def conjugate_p_by_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
        Calculates conjugation q * p * q^-1 of (possibly batched) quaternions p and q
        If p is a vector quaternion representing a position P, and q is a unit quaternion
        representing a rotation R, the result is a vector quaternion representing the new
        position when point P is rotated by R
    """
    q_inv = conjugate_quat(q) / quat_norm(q)**2
    qp = hamilton_product(q, p)
    qpq_inv = hamilton_product(qp, q_inv)
    print(qpq_inv)
 216: conjugate_p_by_q(q[0], q)
 217:
def conjugate_p_by_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
        Calculates conjugation q * p * q^-1 of (possibly batched) quaternions p and q
        If p is a vector quaternion representing a position P, and q is a unit quaternion
        representing a rotation R, the result is a vector quaternion representing the new
        position when point P is rotated by R
    """
    q_inv = conjugate_quat(q) / quat_norm(q)**2
    qp = hamilton_product(q, p)
    qpq_inv = hamilton_product(qp, q_inv)
    return qpq_inv
 218: conjugate_p_by_q(q[0], q)
 219: conjugate_p_by_q(q, q)
 220: np.cos(3/2*np.pi)
 221: np.cos(np.pi/2)
 222:
point = np.asarray([0,1,0,0])
rot = np.asarray([np.cos((3/4)*np.pi), 0, np.sin((3/4)*np.pi), 0])
conjugate_p_by_q(p=point, q=rot)
# should be [0,0,0,1]
 223:
def rot_quat_from_rot_vec(rot_vec: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from the given rotation vector.
        Uses result of Taylor expansion of the exponential function.
    """
    single = False
    if rot_vec.ndim == 1:
        single = True
        rot_vec = rot_vec.reshape(1, -1)
    angles = np.linalg.norm(rot_vec, axis=1).reshape(-1, 1)
    print(angles)
 224:
rot_vec = np.asarray([1,0,0])

rot_quat_from_rot_vec(
 225:
rot_vec = np.asarray([1,0,0])

rot_quat_from_rot_vec(rot_vec)
 226:
rot_vec = np.asarray([1,1,0])
rot_quat_from_rot_vec(rot_vec)
 227:
def rot_quat_from_rot_vec(rot_vec: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from the given rotation vector.
        The rotation angle theta [rad] is determined from the length of the rotation
        vector, while the rotation axis (x,y,z) is given by the normed rotation vector.
        Uses quaternion polar decomposition (via Taylor expansion of the exponential
        function) to determine rotation quaternion:
        q = (cos(theta/2) + x * sin(theta/2)i + y * sin(theta/2)j + z * sin(theta/2)k)
    """
    single = False
    if rot_vec.ndim == 1:
        single = True
        rot_vec = rot_vec.reshape(1, -1)
    angles = np.linalg.norm(rot_vec, axis=1).reshape(-1, 1)
    print(angles)
    euler_axes = rot_vec / angles
    print(euler_axes)
 228:
rot_vec = np.asarray([1,1,0])
rot_quat_from_rot_vec(rot_vec)
 229: np.linalg.norm([0.70710678, 0.70710678, 0.        ])
 230:
def rot_quat_from_rot_vec(rot_vec: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from the given rotation vector.
        The rotation angle theta [rad] is determined from the length of the rotation
        vector, while the rotation axis (x,y,z) is given by the normed rotation vector.
        Uses quaternion polar decomposition (via Taylor expansion of the exponential
        function) to determine rotation quaternion:
        q = exp(theta * rot_axis) [where rot_axis is vector quaternion representing the
                                   rotation axis]
        q = (cos(theta/2) + x * sin(theta/2)i + y * sin(theta/2)j + z * sin(theta/2)k)
    """
    single = False
    if rot_vec.ndim == 1:
        single = True
        rot_vec = rot_vec.reshape(1, -1)
    angles = np.linalg.norm(rot_vec, axis=1).reshape(-1, 1)
    euler_axes = rot_vec / angles
    angles = angles.reshape(-1, 1)
    print(angles)
    print(euler_axes)
    rx = euler_axes[:, 0].reshape(-1, 1)
    ry = euler_axes[:, 1].reshape(-1, 1)
    rz = euler_axes[:, 1].reshape(-1, 1)
    print(rx)
    print(ry)
    print(rz)
    #q = np.concatenate((np.cos(
 231:
rot_vec = np.asarray([1,1,0])
rot_quat_from_rot_vec(rot_vec)
 232:
def rot_quat_from_rot_vec(rot_vec: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from the given rotation vector.
        The rotation angle theta [rad] is determined from the length of the rotation
        vector, while the rotation axis (x,y,z) is given by the normed rotation vector.
        Uses quaternion polar decomposition (via Taylor expansion of the exponential
        function) to determine rotation quaternion:
        q = exp(theta * rot_axis) [where rot_axis is vector quaternion representing the
                                   rotation axis]
        q = (cos(theta/2) + x * sin(theta/2)i + y * sin(theta/2)j + z * sin(theta/2)k)
    """
    single = False
    if rot_vec.ndim == 1:
        single = True
        rot_vec = rot_vec.reshape(1, -1)
    angles = np.linalg.norm(rot_vec, axis=1).reshape(-1, 1)
    euler_axes = rot_vec / angles
    angles = angles.reshape(-1, 1)
    print(angles)
    print(euler_axes)
    rx = euler_axes[:, 0].reshape(-1, 1)
    ry = euler_axes[:, 1].reshape(-1, 1)
    rz = euler_axes[:, 2].reshape(-1, 1)
    print(rx)
    print(ry)
    print(rz)
    #q = np.concatenate((np.cos(
 233:
rot_vec = np.asarray([1,1,0])
rot_quat_from_rot_vec(rot_vec)
 234:
def rot_quat_from_rot_vec(rot_vec: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from the given rotation vector.
        The rotation angle theta [rad] is determined from the length of the rotation
        vector, while the rotation axis (x,y,z) is given by the normed rotation vector.
        Uses quaternion polar decomposition (via Taylor expansion of the exponential
        function) to determine rotation quaternion:
        q = exp(theta * rot_axis) [where rot_axis is vector quaternion representing the
                                   rotation axis]
        q = (cos(theta/2) + x * sin(theta/2)i + y * sin(theta/2)j + z * sin(theta/2)k)
    """
    single = False
    if rot_vec.ndim == 1:
        single = True
        rot_vec = rot_vec.reshape(1, -1)
    angles = np.linalg.norm(rot_vec, axis=1).reshape(-1, 1)
    euler_axes = rot_vec / angles
    angles = angles.reshape(-1, 1)
    print(angles)
    print(euler_axes)
    rx = euler_axes[:, 0].reshape(-1, 1)
    ry = euler_axes[:, 1].reshape(-1, 1)
    rz = euler_axes[:, 2].reshape(-1, 1)
    print(rx)
    print(ry)
    print(rz)
    print(np.cos(angles/2))
    #q = np.concatenate((np.cos(
 235:
rot_vec = np.asarray([1,1,0])
rot_quat_from_rot_vec(rot_vec)
 236:
def rot_quat_from_rot_vec(rot_vec: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from the given rotation vector.
        The rotation angle theta [rad] is determined from the length of the rotation
        vector, while the rotation axis (x,y,z) is given by the normed rotation vector.
        Uses quaternion polar decomposition (via Taylor expansion of the exponential
        function) to determine rotation quaternion:
        q = exp(theta * rot_axis) [where rot_axis is vector quaternion representing the
                                   rotation axis]
        q = (cos(theta/2) + x * sin(theta/2)i + y * sin(theta/2)j + z * sin(theta/2)k)
    """
    single = False
    if rot_vec.ndim == 1:
        single = True
        rot_vec = rot_vec.reshape(1, -1)
    angles = np.linalg.norm(rot_vec, axis=1).reshape(-1, 1)
    euler_axes = rot_vec / angles
    angles = angles.reshape(-1, 1)
    print(angles)
    print(euler_axes)
    rx = euler_axes[:, 0].reshape(-1, 1)
    ry = euler_axes[:, 1].reshape(-1, 1)
    rz = euler_axes[:, 2].reshape(-1, 1)
    print(rx)
    print(ry)
    print(rz)
    print(np.cos(angles/2).reshape(-1, 1))
    #q = np.concatenate((np.cos(
 237:
rot_vec = np.asarray([1,1,0])
rot_quat_from_rot_vec(rot_vec)
 238:
r = np.asarray([1,2,3])
r
 239:
r = np.asarray([1,2,3])
r * np.sin(np.pi/2)
 240:
r = np.asarray([[1,2,3]])
r * np.sin(np.pi/2)
 241:
r = np.asarray([[1,2,3], [4,5,6]])
r * np.sin(np.pi/2)
 242:
rot_vec = np.asarray([[1,1,0], [1,0,0])
rot_quat_from_rot_vec(rot_vec)
 243:
rot_vec = np.asarray([[1,1,0], [1,0,0]])
rot_quat_from_rot_vec(rot_vec)
 244:
rot_vec = np.asarray([[1,1,0], [0,0,0]])
rot_quat_from_rot_vec(rot_vec)
 245:
rot_vec = np.asarray([[1,1,0], [1,0,0]])
rot_quat_from_rot_vec(rot_vec)
 246:
def rot_quat_from_rot_vec(rot_vec: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from the given rotation vector.
        The rotation angle theta [rad] is determined from the length of the rotation
        vector, while the rotation axis (x,y,z) is given by the normed rotation vector.
        Uses quaternion polar decomposition (via Taylor expansion of the exponential
        function) to determine rotation quaternion:
        q = exp(theta * rot_axis) [where rot_axis is vector quaternion representing the
                                   rotation axis]
        q = (cos(theta/2) + x * sin(theta/2)i + y * sin(theta/2)j + z * sin(theta/2)k)
    """
    single = False
    if rot_vec.ndim == 1:
        single = True
        rot_vec = rot_vec.reshape(1, -1)
    angles = np.linalg.norm(rot_vec, axis=1).reshape(-1, 1)
    euler_axes = rot_vec / angles
    angles = angles.reshape(-1, 1)
    print(angles)
    print(euler_axes)
    q = np.concatenate((np.cos(angles/2),
                        euler_axes * np.sin(angles/2)
                       ), axis=1, dtype=np.float64)
    print(q)
 247:
rot_vec = np.asarray([[1,1,0], [1,0,0]])
rot_quat_from_rot_vec(rot_vec)
 248:
rot_vec = np.asarray([[1,1,0], [0,1,0]])
rot_quat_from_rot_vec(rot_vec)
 249:
rot_vec = np.asarray([0,1,0]) * (3/2)*np.pi
rot_quat_from_rot_vec(rot_vec)
 250: -1/np.sqrt(2)
 251:
rot_vec = np.asarray([[0,1,0], [0,1,0]]) * (3/2)*np.pi
rot_quat_from_rot_vec(rot_vec)
 252:
def rot_quat_from_rot_vec(rot_vec: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from the given rotation vector.
        The rotation angle theta [rad] is determined from the length of the rotation
        vector, while the rotation axis (x,y,z) is given by the normed rotation vector.
        Uses quaternion polar decomposition (via Taylor expansion of the exponential
        function) to determine rotation quaternion:
        q = exp(theta * rot_axis) [where rot_axis is vector quaternion representing the
                                   rotation axis]
        q = (cos(theta/2) + x * sin(theta/2)i + y * sin(theta/2)j + z * sin(theta/2)k)
    """
    single = False
    if rot_vec.ndim == 1:
        single = True
        rot_vec = rot_vec.reshape(1, -1)
    angles = np.linalg.norm(rot_vec, axis=1).reshape(-1, 1)
    euler_axes = rot_vec / angles
    angles = angles.reshape(-1, 1)
    print(angles)
    print(euler_axes)
    q = np.concatenate((np.cos(angles/2),
                        euler_axes * np.sin(angles/2)
                       ), axis=1, dtype=np.float64)
    if single:
        return q[0]
    return q
 253:
rot_vec = np.asarray([[0,1,0], [0,1,0]]) * (3/2)*np.pi
rot_quat_from_rot_vec(rot_vec)
 254:
rot_vec = np.asarray([0,1,0]) * (3/2)*np.pi
rot_quat_from_rot_vec(rot_vec)
 255:
def rot_quat_from_rot_vec(rot_vec: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from the given rotation vector.
        The rotation angle theta [rad] is determined from the length of the rotation
        vector, while the rotation axis (x,y,z) is given by the normed rotation vector.
        Uses quaternion polar decomposition (via Taylor expansion of the exponential
        function) to determine rotation quaternion:
        q = exp(theta * rot_axis) [where rot_axis is vector quaternion representing the
                                   rotation axis]
        q = (cos(theta/2) + x * sin(theta/2)i + y * sin(theta/2)j + z * sin(theta/2)k)
    """
    single = False
    if rot_vec.ndim == 1:
        single = True
        rot_vec = rot_vec.reshape(1, -1)
    angles = np.linalg.norm(rot_vec, axis=1).reshape(-1, 1)
    euler_axes = rot_vec / angles
    angles = angles.reshape(-1, 1)
    q = np.concatenate((np.cos(angles/2),
                        euler_axes * np.sin(angles/2)
                       ), axis=1, dtype=np.float64)
    if single:
        return q[0]
    return q
 256:
rot_vec = np.asarray([0,1,0]) * (3/2)*np.pi
rot_quat_from_rot_vec(rot_vec)
 257:
a = np.asarray([[1,2]])
np.tile(a, (2,1))
 258:
def rot_quat_from_rot_vec(rot_vec: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from the given rotation vector.
        The rotation angle theta [rad] is determined from the length of the rotation
        vector, while the rotation axis (x,y,z) is given by the normed rotation vector.
        Uses quaternion polar decomposition (via Taylor expansion of the exponential
        function) to determine rotation quaternion:
        q = exp(theta * rot_axis) [where rot_axis is vector quaternion representing the
                                   rotation axis]
        q = (cos(theta/2) + x * sin(theta/2)i + y * sin(theta/2)j + z * sin(theta/2)k)
    """
    single = False
    if rot_vec.ndim == 1:
        single = True
        rot_vec = rot_vec.reshape(1, -1)
    angles = np.linalg.norm(rot_vec, axis=1).reshape(-1, 1)
    euler_axes = rot_vec / angles
    angles = angles.reshape(-1, 1)
    q = np.concatenate((np.cos(angles/2),
                        euler_axes * np.sin(angles/2)
                       ), axis=1, dtype=np.float64)
    if single:
        return q[0]
    return q

def rot_quat_from_axis_and_angle(axes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from given rotation axes and
        rotation angles. See `rot_quat_from_rot_vec` for details
    """
    axes_single = False
    angles_single = False
    if axes.ndim == 1:
        axes_single = True
        axes = axes.reshape(1, -1)
    if angles.ndim == 1:
        angles_single = True
        angles = angles.reshape(1, -1)
    if axes.shape[0] == 1 and angles.shape[0] > 1:
        axes = np.tile(axes, (angles.shape[0], 1))
    if angles.shape[0] == 1 and axes.shape[0] > 1:
        angles = np.tile(angles, (axes.shape[0], 1))
    assert axes.shape[0] == angles.shape[0], f"Axes and angles have to have the same batch size, got {axes.shape[0]} and {angles.shape[0]}"
    # norm axes in case
    axes = axes / np.linalg.norm(axes, axis=1).reshape(-1 ,1)
    print(angles)
    print(axes)
 259:
rot_vec = np.asarray([0,1,0]) * (3/2)*np.pi
rot_quat_from_rot_vec(rot_vec)
# should be (-1/sqrt(2) + 0i + 1/sqrt(2)j + 0k)
rot_quat_from_axis_and_angle([0,1,0], (3/2)*np.pi)
 260:
rot_vec = np.asarray([0,1,0]) * (3/2)*np.pi
rot_quat_from_rot_vec(rot_vec)
# should be (-1/sqrt(2) + 0i + 1/sqrt(2)j + 0k)
rot_quat_from_axis_and_angle(np.asarray([0,1,0]), np.asarray((3/2)*np.pi))
 261:
rot_vec = np.asarray([0,1,0]) * (3/2)*np.pi
rot_quat_from_rot_vec(rot_vec)
# should be (-1/sqrt(2) + 0i + 1/sqrt(2)j + 0k)
rot_quat_from_axis_and_angle(np.asarray([0,1,0]), np.asarray([(3/2)*np.pi])
 262:
rot_vec = np.asarray([0,1,0]) * (3/2)*np.pi
rot_quat_from_rot_vec(rot_vec)
# should be (-1/sqrt(2) + 0i + 1/sqrt(2)j + 0k)
rot_quat_from_axis_and_angle(np.asarray([0,1,0]), np.asarray([(3/2)*np.pi]))
 263:
rot_vec = np.asarray([0,1,0]) * (3/2)*np.pi
print(rot_quat_from_rot_vec(rot_vec))
# should be (-1/sqrt(2) + 0i + 1/sqrt(2)j + 0k)
rot_quat_from_axis_and_angle(np.asarray([0,1,0]), np.asarray([(3/2)*np.pi]))
 264: (3/2)*np.pi
 265:
def rot_quat_from_rot_vec(rot_vec: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from the given rotation vector.
        The rotation angle theta [rad] is determined from the length of the rotation
        vector, while the rotation axis (x,y,z) is given by the normed rotation vector.
        Uses quaternion polar decomposition (via Taylor expansion of the exponential
        function) to determine rotation quaternion:
        q = exp(theta * rot_axis) [where rot_axis is vector quaternion representing the
                                   rotation axis]
        q = (cos(theta/2) + x * sin(theta/2)i + y * sin(theta/2)j + z * sin(theta/2)k)
    """
    single = False
    if rot_vec.ndim == 1:
        single = True
        rot_vec = rot_vec.reshape(1, -1)
    angles = np.linalg.norm(rot_vec, axis=1).reshape(-1, 1)
    euler_axes = rot_vec / angles
    angles = angles.reshape(-1, 1)
    q = np.concatenate((np.cos(angles/2),
                        euler_axes * np.sin(angles/2)
                       ), axis=1, dtype=np.float64)
    if single:
        return q[0]
    return q

def rot_quat_from_axis_and_angle(axes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from given rotation axes and
        rotation angles. See `rot_quat_from_rot_vec` for details
    """
    axes_single = False
    angles_single = False
    if axes.ndim == 1:
        axes_single = True
        axes = axes.reshape(1, -1)
    if angles.ndim == 1:
        angles_single = True
        angles = angles.reshape(1, -1)
    if axes.shape[0] == 1 and angles.shape[0] > 1:
        axes = np.tile(axes, (angles.shape[0], 1))
    if angles.shape[0] == 1 and axes.shape[0] > 1:
        angles = np.tile(angles, (axes.shape[0], 1))
    assert axes.shape[0] == angles.shape[0], f"Axes and angles have to have the same batch size, got {axes.shape[0]} and {angles.shape[0]}"
    # norm axes in case
    axes = axes / np.linalg.norm(axes, axis=1).reshape(-1 ,1)
    q = np.concatenate((np.cos(angles/2),
                        axes * np.sin(angels/2)
                       ), axis=1, dtype=np.float64)
    if axes_single and angles_single:
        return q[0]
    return q
 266:
rot_vec = np.asarray([0,1,0]) * (3/2)*np.pi
print(rot_quat_from_rot_vec(rot_vec))
# should be (-1/sqrt(2) + 0i + 1/sqrt(2)j + 0k)
rot_quat_from_axis_and_angle(np.asarray([0,1,0]), np.asarray([(3/2)*np.pi]))
 267:
def rot_quat_from_rot_vec(rot_vec: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from the given rotation vector.
        The rotation angle theta [rad] is determined from the length of the rotation
        vector, while the rotation axis (x,y,z) is given by the normed rotation vector.
        Uses quaternion polar decomposition (via Taylor expansion of the exponential
        function) to determine rotation quaternion:
        q = exp(theta * rot_axis) [where rot_axis is vector quaternion representing the
                                   rotation axis]
        q = (cos(theta/2) + x * sin(theta/2)i + y * sin(theta/2)j + z * sin(theta/2)k)
    """
    single = False
    if rot_vec.ndim == 1:
        single = True
        rot_vec = rot_vec.reshape(1, -1)
    angles = np.linalg.norm(rot_vec, axis=1).reshape(-1, 1)
    euler_axes = rot_vec / angles
    angles = angles.reshape(-1, 1)
    q = np.concatenate((np.cos(angles/2),
                        euler_axes * np.sin(angles/2)
                       ), axis=1, dtype=np.float64)
    if single:
        return q[0]
    return q

def rot_quat_from_axis_and_angle(axes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from given rotation axes and
        rotation angles. See `rot_quat_from_rot_vec` for details
    """
    axes_single = False
    angles_single = False
    if axes.ndim == 1:
        axes_single = True
        axes = axes.reshape(1, -1)
    if angles.ndim == 1:
        angles_single = True
        angles = angles.reshape(1, -1)
    if axes.shape[0] == 1 and angles.shape[0] > 1:
        axes = np.tile(axes, (angles.shape[0], 1))
    if angles.shape[0] == 1 and axes.shape[0] > 1:
        angles = np.tile(angles, (axes.shape[0], 1))
    assert axes.shape[0] == angles.shape[0], f"Axes and angles have to have the same batch size, got {axes.shape[0]} and {angles.shape[0]}"
    # norm axes in case
    axes = axes / np.linalg.norm(axes, axis=1).reshape(-1 ,1)
    q = np.concatenate((np.cos(angles/2),
                        axes * np.sin(angles/2)
                       ), axis=1, dtype=np.float64)
    if axes_single and angles_single:
        return q[0]
    return q
 268:
rot_vec = np.asarray([0,1,0]) * (3/2)*np.pi
print(rot_quat_from_rot_vec(rot_vec))
# should be (-1/sqrt(2) + 0i + 1/sqrt(2)j + 0k)
rot_quat_from_axis_and_angle(np.asarray([0,1,0]), np.asarray([(3/2)*np.pi]))
 269:
def rot_quat_from_rot_vec(rot_vec: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from the given rotation vector.
        The rotation angle theta [rad] is determined from the length of the rotation
        vector, while the rotation axis (x,y,z) is given by the normed rotation vector.
        Uses quaternion polar decomposition (via Taylor expansion of the exponential
        function) to determine rotation quaternion:
        q = exp(theta * rot_axis) [where rot_axis is vector quaternion representing the
                                   rotation axis]
        q = (cos(theta/2) + x * sin(theta/2)i + y * sin(theta/2)j + z * sin(theta/2)k)
    """
    single = False
    if rot_vec.ndim == 1:
        single = True
        rot_vec = rot_vec.reshape(1, -1)
    angles = np.linalg.norm(rot_vec, axis=1).reshape(-1, 1)
    euler_axes = rot_vec / angles
    angles = angles.reshape(-1, 1)
    q = np.concatenate((np.cos(angles/2),
                        euler_axes * np.sin(angles/2)
                       ), axis=1, dtype=np.float64)
    if single:
        return q[0]
    return q

def rot_quat_from_axis_and_angle(axes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from given rotation axes and
        rotation angles. See `rot_quat_from_rot_vec` for details
    """
    axes_single = False
    angles_single = False
    if axes.ndim == 1:
        axes_single = True
        axes = axes.reshape(1, -1)
    if angles.ndim == 1:
        angles_single = True
        angles = angles.reshape(1, -1)
    if axes.shape[0] == 1 and angles.shape[0] > 1:
        axes = np.tile(axes, (angles.shape[0], 1))
    if angles.shape[0] == 1 and axes.shape[0] > 1:
        angles = np.tile(angles, (axes.shape[0], 1))
    assert axes.shape[0] == angles.shape[0], f"Axes and angles have to have the same batch size, got {axes.shape[0]} and {angles.shape[0]}"
    # norm axes in case
    axes = axes / np.linalg.norm(axes, axis=1).reshape(-1 ,1)
    q = np.concatenate((np.cos(angles/2),
                        axes * np.sin(angles/2)
                       ), axis=1, dtype=np.float64)
    if axes_single and angles_single:
        return q[0]
    return q

def point_as_quat(p: np.ndarray) -> np.ndarray:
    """
        Represents point p=(x,y,z) as a vector quaternion
        p' = 0 + xi + yj + zk
    """
    single = False
    if p.ndim == 1:
        single = True
        p = p.reshape(1, -1)
    q = np.concatenate((np.zeros((p.shape[0], 0), dtype=np.float64),
                        p[:, 0].reshape(-1, 1),
                        p[:, 1].reshape(-1, 1),
                        p[:, 2].reshape(-1, 1)
                       ), axis=1, dtype=np.float64)
    if single:
        return q[0]
    return q
 270:
rot_vec = np.asarray([0,1,0]) * (3/2)*np.pi
print(rot_quat_from_rot_vec(rot_vec))
# should be (-1/sqrt(2) + 0i + 1/sqrt(2)j + 0k)
print(rot_quat_from_axis_and_angle(np.asarray([0,1,0]), np.asarray([(3/2)*np.pi])))
point_as_quat(np.asarray([1,2,3]))
 271:
def rot_quat_from_rot_vec(rot_vec: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from the given rotation vector.
        The rotation angle theta [rad] is determined from the length of the rotation
        vector, while the rotation axis (x,y,z) is given by the normed rotation vector.
        Uses quaternion polar decomposition (via Taylor expansion of the exponential
        function) to determine rotation quaternion:
        q = exp(theta * rot_axis) [where rot_axis is vector quaternion representing the
                                   rotation axis]
        q = (cos(theta/2) + x * sin(theta/2)i + y * sin(theta/2)j + z * sin(theta/2)k)
    """
    single = False
    if rot_vec.ndim == 1:
        single = True
        rot_vec = rot_vec.reshape(1, -1)
    angles = np.linalg.norm(rot_vec, axis=1).reshape(-1, 1)
    euler_axes = rot_vec / angles
    angles = angles.reshape(-1, 1)
    q = np.concatenate((np.cos(angles/2),
                        euler_axes * np.sin(angles/2)
                       ), axis=1, dtype=np.float64)
    if single:
        return q[0]
    return q

def rot_quat_from_axis_and_angle(axes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from given rotation axes and
        rotation angles. See `rot_quat_from_rot_vec` for details
    """
    axes_single = False
    angles_single = False
    if axes.ndim == 1:
        axes_single = True
        axes = axes.reshape(1, -1)
    if angles.ndim == 1:
        angles_single = True
        angles = angles.reshape(1, -1)
    if axes.shape[0] == 1 and angles.shape[0] > 1:
        axes = np.tile(axes, (angles.shape[0], 1))
    if angles.shape[0] == 1 and axes.shape[0] > 1:
        angles = np.tile(angles, (axes.shape[0], 1))
    assert axes.shape[0] == angles.shape[0], f"Axes and angles have to have the same batch size, got {axes.shape[0]} and {angles.shape[0]}"
    # norm axes in case
    axes = axes / np.linalg.norm(axes, axis=1).reshape(-1 ,1)
    q = np.concatenate((np.cos(angles/2),
                        axes * np.sin(angles/2)
                       ), axis=1, dtype=np.float64)
    if axes_single and angles_single:
        return q[0]
    return q

def point_as_quat(p: np.ndarray) -> np.ndarray:
    """
        Represents point p=(x,y,z) as a vector quaternion
        p' = 0 + xi + yj + zk
    """
    single = False
    if p.ndim == 1:
        single = True
        p = p.reshape(1, -1)
    q = np.concatenate((np.zeros((p.shape[0], 0), dtype=np.float64),
                        p[:, 0].reshape(-1, 1),
                        p[:, 1].reshape(-1, 1),
                        p[:, 2].reshape(-1, 1)
                       ), axis=1, dtype=np.float64)
    if single:
        return q[0]
    return q
 272:
rot_vec = np.asarray([0,1,0]) * (3/2)*np.pi
print(rot_quat_from_rot_vec(rot_vec))
# should be (-1/sqrt(2) + 0i + 1/sqrt(2)j + 0k)
print(rot_quat_from_axis_and_angle(np.asarray([0,1,0]), np.asarray([(3/2)*np.pi])))
point_as_quat(np.asarray([1,2,3]))
 273:
def rot_quat_from_rot_vec(rot_vec: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from the given rotation vector.
        The rotation angle theta [rad] is determined from the length of the rotation
        vector, while the rotation axis (x,y,z) is given by the normed rotation vector.
        Uses quaternion polar decomposition (via Taylor expansion of the exponential
        function) to determine rotation quaternion:
        q = exp(theta * rot_axis) [where rot_axis is vector quaternion representing the
                                   rotation axis]
        q = (cos(theta/2) + x * sin(theta/2)i + y * sin(theta/2)j + z * sin(theta/2)k)
    """
    single = False
    if rot_vec.ndim == 1:
        single = True
        rot_vec = rot_vec.reshape(1, -1)
    angles = np.linalg.norm(rot_vec, axis=1).reshape(-1, 1)
    euler_axes = rot_vec / angles
    angles = angles.reshape(-1, 1)
    q = np.concatenate((np.cos(angles/2),
                        euler_axes * np.sin(angles/2)
                       ), axis=1, dtype=np.float64)
    if single:
        return q[0]
    return q

def rot_quat_from_axis_and_angle(axes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from given rotation axes and
        rotation angles. See `rot_quat_from_rot_vec` for details
    """
    axes_single = False
    angles_single = False
    if axes.ndim == 1:
        axes_single = True
        axes = axes.reshape(1, -1)
    if angles.ndim == 1:
        angles_single = True
        angles = angles.reshape(1, -1)
    if axes.shape[0] == 1 and angles.shape[0] > 1:
        axes = np.tile(axes, (angles.shape[0], 1))
    if angles.shape[0] == 1 and axes.shape[0] > 1:
        angles = np.tile(angles, (axes.shape[0], 1))
    assert axes.shape[0] == angles.shape[0], f"Axes and angles have to have the same batch size, got {axes.shape[0]} and {angles.shape[0]}"
    # norm axes in case
    axes = axes / np.linalg.norm(axes, axis=1).reshape(-1 ,1)
    q = np.concatenate((np.cos(angles/2),
                        axes * np.sin(angles/2)
                       ), axis=1, dtype=np.float64)
    if axes_single and angles_single:
        return q[0]
    return q

def point_as_quat(p: np.ndarray) -> np.ndarray:
    """
        Represents point p=(x,y,z) as a vector quaternion
        p' = 0 + xi + yj + zk
    """
    single = False
    if p.ndim == 1:
        single = True
        p = p.reshape(1, -1)
    q = np.concatenate((np.zeros((p.shape[0], 1), dtype=np.float64),
                        p[:, 0].reshape(-1, 1),
                        p[:, 1].reshape(-1, 1),
                        p[:, 2].reshape(-1, 1)
                       ), axis=1, dtype=np.float64)
    if single:
        return q[0]
    return q
 274:
rot_vec = np.asarray([0,1,0]) * (3/2)*np.pi
print(rot_quat_from_rot_vec(rot_vec))
# should be (-1/sqrt(2) + 0i + 1/sqrt(2)j + 0k)
print(rot_quat_from_axis_and_angle(np.asarray([0,1,0]), np.asarray([(3/2)*np.pi])))
point_as_quat(np.asarray([1,2,3]))
 275:
rot_vec = np.asarray([0,1,0]) * (3/2)*np.pi
print(rot_quat_from_rot_vec(rot_vec))
# should be (-1/sqrt(2) + 0i + 1/sqrt(2)j + 0k)
print(rot_quat_from_axis_and_angle(np.asarray([0,1,0]), np.asarray([(3/2)*np.pi])))
point_as_quat(np.asarray([[1,2,3], [2,3,4]]))
 276:
def rot_quat_from_rot_vec(rot_vec: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from the given rotation vector.
        The rotation angle theta [rad] is determined from the length of the rotation
        vector, while the rotation axis (x,y,z) is given by the normed rotation vector.
        Uses quaternion polar decomposition (via Taylor expansion of the exponential
        function) to determine rotation quaternion:
        q = exp(theta * rot_axis) [where rot_axis is vector quaternion representing the
                                   rotation axis]
        q = (cos(theta/2) + x * sin(theta/2)i + y * sin(theta/2)j + z * sin(theta/2)k)
    """
    single = False
    if rot_vec.ndim == 1:
        single = True
        rot_vec = rot_vec.reshape(1, -1)
    angles = np.linalg.norm(rot_vec, axis=1).reshape(-1, 1)
    euler_axes = rot_vec / angles
    angles = angles.reshape(-1, 1)
    q = np.concatenate((np.cos(angles/2),
                        euler_axes * np.sin(angles/2)
                       ), axis=1, dtype=np.float64)
    if single:
        return q[0]
    return q

def rot_quat_from_axis_and_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """
        Calculates a rotation quaternion (i.e. versor) from given rotation axis and
        rotation angle. See `rot_quat_from_rot_vec` for details
    """
    axis_single = False
    angle_single = False
    if axis.ndim == 1:
        axis_single = True
        axis = axis.reshape(1, -1)
    if angle.ndim == 1:
        angle_single = True
        angle = angle.reshape(1, -1)
    if axis.shape[0] == 1 and angle.shape[0] > 1:
        axis = np.tile(axis, (angle.shape[0], 1))
    if angle.shape[0] == 1 and axis.shape[0] > 1:
        angle = np.tile(angle, (axis.shape[0], 1))
    assert axis.shape[0] == angle.shape[0], f"Axes and angles have to have the same batch size, got {axis.shape[0]} and {angle.shape[0]}"
    # norm axes in case
    axis = axis / np.linalg.norm(axis, axis=1).reshape(-1 ,1)
    q = np.concatenate((np.cos(angle/2),
                        axis * np.sin(angle/2)
                       ), axis=1, dtype=np.float64)
    if axis_single and angle_single:
        return q[0]
    return q

def point_as_quat(p: np.ndarray) -> np.ndarray:
    """
        Represents point p=(x,y,z) as a vector quaternion
        p' = 0 + xi + yj + zk
    """
    single = False
    if p.ndim == 1:
        single = True
        p = p.reshape(1, -1)
    q = np.concatenate((np.zeros((p.shape[0], 1), dtype=np.float64),
                        p[:, 0].reshape(-1, 1),
                        p[:, 1].reshape(-1, 1),
                        p[:, 2].reshape(-1, 1)
                       ), axis=1, dtype=np.float64)
    if single:
        return q[0]
    return q
 277:
rot_vec = np.asarray([0,1,0]) * (3/2)*np.pi
print(rot_quat_from_rot_vec(rot_vec))
# should be (-1/sqrt(2) + 0i + 1/sqrt(2)j + 0k)
print(rot_quat_from_axis_and_angle(np.asarray([0,1,0]), np.asarray([(3/2)*np.pi])))
point_as_quat(np.asarray([[1,2,3], [2,3,4]]))
 278:
point = point_as_quat(np.asarray([1,0,0]))
rot = rot_quat_from_axis_and_angle(axis=np.asarray([0,1,0]), angle=np.asarray([(3/2)*np.pi]))
conjugate_p_by_q(p=point, q=rot)
# should be [0,0,0,1]
 279:
q = np.asarray([[1,1,1,0], [1,0,0,0], [1,0,1,1]])
print(q)
 280:
def quat_norm(q: np.ndarray) -> float:
    single = False
    if q.ndim == 1:
        single = True
        q = q.reshape(1, -1)
    norm = np.linalg.norm(q, axis=1).reshape(-1, 1)
    if single:
        return norm[0]
    return norm
 281: quat_norm(q)
 282:
def normalize_quat(q: np.ndarray) -> np.ndarray:
    """
        Normalizes (possibly batched) quaternion q
    """
    single_dim = False
    if q.ndim == 1:
        single_dim = True
        q = q.reshape(1, -1)
    q_norm = q / quat_norm(q)
    if single_dim:
        return q_norm[0]
    else:
        return q_norm
 283: normalize_quat(q)
 284:
def conjugate_quat(q: np.ndarray) -> np.ndarray:
    """
        Returns the conjugate of q=(a+bi+cj+dk) as q*=(a-bi-cj-dk)
    """
    single = False
    if q.ndim == 1:
        single = True
        q = q.reshape(1, -1)
    conjugate = np.concatenate((q[:, 0].reshape(-1, 1),
                                -q[:, 1].reshape(-1, 1),
                                -q[:, 2].reshape(-1, 1),
                                -q[:, 3].reshape(-1, 1)
                               ), axis=1, dtype=np.float64)
    if single:
        return conjugate[0]
    return conjugate
 285: conjugate_quat(q)
 286:
def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
        Calculates the Hamilton product of (possibly batched) quaternions q and p
    """
    q_single = False
    p_single = False
    if q.ndim == 1:
        q_single = True
        q = q.reshape(1, -1)
    if p.ndim == 1:
        p_single = True
        p = p.reshape(1, -1)
    if q.shape[0] == 1 and p.shape[0] > 1:
        q = np.tile(q, (p.shape[0], 1))
    if p.shape[0] == 1 and q.shape[0] > 1:
        p = np.tile(p, (q.shape[0], 1))
    assert q.shape[0] == p.shape[0], f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
    assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
    qa = q[:, 0].reshape(-1, 1)
    qb = q[:, 1].reshape(-1, 1)
    qc = q[:, 2].reshape(-1, 1)
    qd = q[:, 3].reshape(-1, 1)
    pa = p[:, 0].reshape(-1, 1)
    pb = p[:, 1].reshape(-1, 1)
    pc = p[:, 2].reshape(-1, 1)
    pd = p[:, 3].reshape(-1, 1)
    res = np.concatenate(
        [
            qa*pa - qb*pb - qc*pc - qd*pd,
            qa*pb + qb*pa + qc*pd - qd*pc,
            qa*pc - qb*pd + qc*pa + qd*pb,
            qa*pd + qb*pc - qc*pb + qd*pa
        ], axis=1, dtype=np.float64)
    if q_single and p_single:
        return res[0]
    return res
 287: hamilton_product(q[0], q)
 288:
def conjugate_p_by_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
        Calculates conjugation q * p * q^-1 of (possibly batched) quaternions p and q
        If p is a vector quaternion representing a position P, and q is a unit quaternion
        representing a rotation R, the result is a vector quaternion representing the new
        position when point P is rotated by R
    """
    q_inv = conjugate_quat(q) / quat_norm(q)**2
    qp = hamilton_product(q, p)
    qpq_inv = hamilton_product(qp, q_inv)
    return qpq_inv

def rotate_point_by_quat(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
        Rotates point p (represented as pure quaternion) by rotation
        quaternion q. Alias for `conjugate_p_by_q`
    """
    return conjugate_p_by_q(p, q)
 289:
def conjugate_p_by_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
        Calculates conjugation q * p * q^-1 of (possibly batched) quaternions p and q
        If p is a vector quaternion representing a position P, and q is a unit quaternion
        representing a rotation R, the result is a vector quaternion representing the new
        position when point P is rotated by R
    """
    q_inv = conjugate_quat(q) / quat_norm(q)**2
    qp = hamilton_product(q, p)
    qpq_inv = hamilton_product(qp, q_inv)
    return qpq_inv

def rotate_point_using_quat(p: np.ndarray, rot: np.ndarray, rot_angle: np.ndarray = None) -> np.ndarray:
    """
        Rotates 3D point p by rotation rot. If rot_angle is None, assumes rot to be
        either a rotation vector (3) or a rotation quaternion (4) based on its length.
        If rot_angle is given, assumes rot to be an euler angle (does not necessarily
        needs to be a unit vector).
    """
    if (rot.ndim == 1 and rot.shape[0] == 4) or (rot.ndim == 2 and rot.shape[1] == 4):
        rot_quat = rot
    else:
        if rot_angle is None:
            rot_quat = rot_quat_from_rot_vec(rot)
        else:
            rot_quat = rot_quat_from_axis_and_angle(rot, rot_angle)
    point = point_as_quat(p)
    rot_point = conjugate_p_by_q(p=point, q=rot_quat)
    if rot_point.ndim == 2:
        return rot_point[:, 1:]
    else:
        return rot_point[1:]
 290:
point = point_as_quat(np.asarray([1,0,0]))
rot = rot_quat_from_axis_and_angle(axis=np.asarray([0,1,0]), angle=np.asarray([(3/2)*np.pi]))
print(conjugate_p_by_q(p=point, q=rot))
rotate_point_using_quat(p=point, rot=np.asarray([0,1,0]), rot_angle=np.asarray([(3/2)*np.pi]))
# should be [0,0,0,1]
 291:
def conjugate_p_by_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
        Calculates conjugation q * p * q^-1 of (possibly batched) quaternions p and q
        If p is a vector quaternion representing a position P, and q is a unit quaternion
        representing a rotation R, the result is a vector quaternion representing the new
        position when point P is rotated by R
    """
    q_inv = conjugate_quat(q) / quat_norm(q)**2
    qp = hamilton_product(q, p)
    qpq_inv = hamilton_product(qp, q_inv)
    return qpq_inv

def rotate_point_using_quat(p: np.ndarray, rot: np.ndarray, rot_angle: np.ndarray = None) -> np.ndarray:
    """
        Rotates 3D point p by rotation rot. If rot_angle is None, assumes rot to be
        either a rotation vector (3) or a rotation quaternion (4) based on its length.
        If rot_angle is given, assumes rot to be an euler angle (does not necessarily
        needs to be a unit vector).
    """
    if (rot.ndim == 1 and rot.shape[0] == 4) or (rot.ndim == 2 and rot.shape[1] == 4):
        rot_quat = rot
    else:
        if rot_angle is None:
            rot_quat = rot_quat_from_rot_vec(rot)
        else:
            rot_quat = rot_quat_from_axis_and_angle(rot, rot_angle)
    point = point_as_quat(p)
    rot_point = conjugate_p_by_q(p=point, q=rot_quat)
    if rot_point.ndim == 2:
        return rot_point[:, 1:]
    else:
        return rot_point
 292:
point = point_as_quat(np.asarray([1,0,0]))
rot = rot_quat_from_axis_and_angle(axis=np.asarray([0,1,0]), angle=np.asarray([(3/2)*np.pi]))
print(conjugate_p_by_q(p=point, q=rot))
rotate_point_using_quat(p=point, rot=np.asarray([0,1,0]), rot_angle=np.asarray([(3/2)*np.pi]))
# should be [0,0,0,1]
 293:
def conjugate_p_by_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
        Calculates conjugation q * p * q^-1 of (possibly batched) quaternions p and q
        If p is a vector quaternion representing a position P, and q is a unit quaternion
        representing a rotation R, the result is a vector quaternion representing the new
        position when point P is rotated by R
    """
    q_inv = conjugate_quat(q) / quat_norm(q)**2
    qp = hamilton_product(q, p)
    qpq_inv = hamilton_product(qp, q_inv)
    return qpq_inv

def rotate_point_using_quat(p: np.ndarray, rot: np.ndarray, rot_angle: np.ndarray = None) -> np.ndarray:
    """
        Rotates 3D point p by rotation rot. If rot_angle is None, assumes rot to be
        either a rotation vector (3) or a rotation quaternion (4) based on its length.
        If rot_angle is given, assumes rot to be an euler angle (does not necessarily
        needs to be a unit vector).
    """
    if (rot.ndim == 1 and rot.shape[0] == 4) or (rot.ndim == 2 and rot.shape[1] == 4):
        rot_quat = rot
    else:
        if rot_angle is None:
            rot_quat = rot_quat_from_rot_vec(rot)
        else:
            rot_quat = rot_quat_from_axis_and_angle(rot, rot_angle)
    point = point_as_quat(p)
    rot_point = conjugate_p_by_q(p=point, q=rot_quat)
    if rot_point.ndim == 2:
        return rot_point[:, 1:]
    else:
        return rot_point[1:]
 294:
point = point_as_quat(np.asarray([1,0,0]))
rot = rot_quat_from_axis_and_angle(axis=np.asarray([0,1,0]), angle=np.asarray([(3/2)*np.pi]))
print(conjugate_p_by_q(p=point, q=rot))
rotate_point_using_quat(p=point, rot=np.asarray([0,1,0]), rot_angle=np.asarray([(3/2)*np.pi]))
# should be [0,0,0,1]
 295:
def conjugate_p_by_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
        Calculates conjugation q * p * q^-1 of (possibly batched) quaternions p and q
        If p is a vector quaternion representing a position P, and q is a unit quaternion
        representing a rotation R, the result is a vector quaternion representing the new
        position when point P is rotated by R
    """
    q_inv = conjugate_quat(q) / quat_norm(q)**2
    qp = hamilton_product(q, p)
    qpq_inv = hamilton_product(qp, q_inv)
    return qpq_inv

def rotate_point_using_quat(p: np.ndarray, rot: np.ndarray, rot_angle: np.ndarray = None) -> np.ndarray:
    """
        Rotates 3D point p by rotation rot. If rot_angle is None, assumes rot to be
        either a rotation vector (3) or a rotation quaternion (4) based on its length.
        If rot_angle is given, assumes rot to be an euler angle (does not necessarily
        needs to be a unit vector).
    """
    if (rot.ndim == 1 and rot.shape[0] == 4) or (rot.ndim == 2 and rot.shape[1] == 4):
        rot_quat = rot
    else:
        if rot_angle is None:
            rot_quat = rot_quat_from_rot_vec(rot)
        else:
            rot_quat = rot_quat_from_axis_and_angle(rot, rot_angle)
    print(rot_quat)
    point = point_as_quat(p)
    rot_point = conjugate_p_by_q(p=point, q=rot_quat)
    if rot_point.ndim == 2:
        return rot_point[:, 1:]
    else:
        return rot_point[1:]
 296:
point = point_as_quat(np.asarray([1,0,0]))
rot = rot_quat_from_axis_and_angle(axis=np.asarray([0,1,0]), angle=np.asarray([(3/2)*np.pi]))
print(conjugate_p_by_q(p=point, q=rot))
rotate_point_using_quat(p=point, rot=np.asarray([0,1,0]), rot_angle=np.asarray([(3/2)*np.pi]))
# should be [0,0,0,1]
 297:
point = point_as_quat(np.asarray([1,0,0]))
rot = rot_quat_from_axis_and_angle(axis=np.asarray([0,1,0]), angle=np.asarray([(3/2)*np.pi]))
print(rot)
print(conjugate_p_by_q(p=point, q=rot))
rotate_point_using_quat(p=point, rot=np.asarray([0,1,0]), rot_angle=np.asarray([(3/2)*np.pi]))
# should be [0,0,0,1]
 298:
def conjugate_p_by_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
        Calculates conjugation q * p * q^-1 of (possibly batched) quaternions p and q
        If p is a vector quaternion representing a position P, and q is a unit quaternion
        representing a rotation R, the result is a vector quaternion representing the new
        position when point P is rotated by R
    """
    q_inv = conjugate_quat(q) / quat_norm(q)**2
    qp = hamilton_product(q, p)
    qpq_inv = hamilton_product(qp, q_inv)
    return qpq_inv

def rotate_point_using_quat(p: np.ndarray, rot: np.ndarray, rot_angle: np.ndarray = None) -> np.ndarray:
    """
        Rotates 3D point p by rotation rot. If rot_angle is None, assumes rot to be
        either a rotation vector (3) or a rotation quaternion (4) based on its length.
        If rot_angle is given, assumes rot to be an euler angle (does not necessarily
        needs to be a unit vector).
    """
    if (rot.ndim == 1 and rot.shape[0] == 4) or (rot.ndim == 2 and rot.shape[1] == 4):
        rot_quat = rot
    else:
        if rot_angle is None:
            rot_quat = rot_quat_from_rot_vec(rot)
        else:
            rot_quat = rot_quat_from_axis_and_angle(rot, rot_angle)
    point = point_as_quat(p)
    print(point)
    rot_point = conjugate_p_by_q(p=point, q=rot_quat)
    if rot_point.ndim == 2:
        return rot_point[:, 1:]
    else:
        return rot_point[1:]
 299:
point = point_as_quat(np.asarray([1,0,0]))
rot = rot_quat_from_axis_and_angle(axis=np.asarray([0,1,0]), angle=np.asarray([(3/2)*np.pi]))
print(conjugate_p_by_q(p=point, q=rot))
rotate_point_using_quat(p=point, rot=np.asarray([0,1,0]), rot_angle=np.asarray([(3/2)*np.pi]))
# should be [0,0,0,1]
 300:
point = point_as_quat(np.asarray([1,0,0]))
print(point)
rot = rot_quat_from_axis_and_angle(axis=np.asarray([0,1,0]), angle=np.asarray([(3/2)*np.pi]))
print(conjugate_p_by_q(p=point, q=rot))
rotate_point_using_quat(p=point, rot=np.asarray([0,1,0]), rot_angle=np.asarray([(3/2)*np.pi]))
# should be [0,0,0,1]
 301:
point = point_as_quat(np.asarray([1,0,0]))
print(point)
rot = rot_quat_from_axis_and_angle(axis=np.asarray([0,1,0]), angle=np.asarray([(3/2)*np.pi]))
print(conjugate_p_by_q(p=point, q=rot))
print(point)
rotate_point_using_quat(p=point, rot=np.asarray([0,1,0]), rot_angle=np.asarray([(3/2)*np.pi]))
# should be [0,0,0,1]
 302:
def conjugate_p_by_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
        Calculates conjugation q * p * q^-1 of (possibly batched) quaternions p and q
        If p is a vector quaternion representing a position P, and q is a unit quaternion
        representing a rotation R, the result is a vector quaternion representing the new
        position when point P is rotated by R
    """
    q_inv = conjugate_quat(q) / quat_norm(q)**2
    qp = hamilton_product(q, p)
    qpq_inv = hamilton_product(qp, q_inv)
    return qpq_inv

def rotate_point_using_quat(p: np.ndarray, rot: np.ndarray, rot_angle: np.ndarray = None) -> np.ndarray:
    """
        Rotates 3D point p by rotation rot. If rot_angle is None, assumes rot to be
        either a rotation vector (3) or a rotation quaternion (4) based on its length.
        If rot_angle is given, assumes rot to be an euler angle (does not necessarily
        needs to be a unit vector).
    """
    if (rot.ndim == 1 and rot.shape[0] == 4) or (rot.ndim == 2 and rot.shape[1] == 4):
        rot_quat = rot
    else:
        if rot_angle is None:
            rot_quat = rot_quat_from_rot_vec(rot)
        else:
            rot_quat = rot_quat_from_axis_and_angle(rot, rot_angle)
    print(p)
    point = point_as_quat(p)
    print(point)
    rot_point = conjugate_p_by_q(p=point, q=rot_quat)
    if rot_point.ndim == 2:
        return rot_point[:, 1:]
    else:
        return rot_point[1:]
 303:
point = point_as_quat(np.asarray([1,0,0]))
print(point)
rot = rot_quat_from_axis_and_angle(axis=np.asarray([0,1,0]), angle=np.asarray([(3/2)*np.pi]))
print(conjugate_p_by_q(p=point, q=rot))
print(point)
rotate_point_using_quat(p=point, rot=np.asarray([0,1,0]), rot_angle=np.asarray([(3/2)*np.pi]))
# should be [0,0,0,1]
 304:
def conjugate_p_by_q(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
        Calculates conjugation q * p * q^-1 of (possibly batched) quaternions p and q
        If p is a vector quaternion representing a position P, and q is a unit quaternion
        representing a rotation R, the result is a vector quaternion representing the new
        position when point P is rotated by R
    """
    q_inv = conjugate_quat(q) / quat_norm(q)**2
    qp = hamilton_product(q, p)
    qpq_inv = hamilton_product(qp, q_inv)
    return qpq_inv

def rotate_point_using_quat(p: np.ndarray, rot: np.ndarray, rot_angle: np.ndarray = None) -> np.ndarray:
    """
        Rotates 3D point p by rotation rot. If rot_angle is None, assumes rot to be
        either a rotation vector (3) or a rotation quaternion (4) based on its length.
        If rot_angle is given, assumes rot to be an euler angle (does not necessarily
        needs to be a unit vector).
    """
    if (rot.ndim == 1 and rot.shape[0] == 4) or (rot.ndim == 2 and rot.shape[1] == 4):
        rot_quat = rot
    else:
        if rot_angle is None:
            rot_quat = rot_quat_from_rot_vec(rot)
        else:
            rot_quat = rot_quat_from_axis_and_angle(rot, rot_angle)
    point = point_as_quat(p)
    rot_point = conjugate_p_by_q(p=point, q=rot_quat)
    if rot_point.ndim == 2:
        return rot_point[:, 1:]
    else:
        return rot_point[1:]
 305:
point = point_as_quat(np.asarray([1,0,0]))
print(point)
rot = rot_quat_from_axis_and_angle(axis=np.asarray([0,1,0]), angle=np.asarray([(3/2)*np.pi]))
print(conjugate_p_by_q(p=point, q=rot))
print(point)
rotate_point_using_quat(p=np.asarray([1,0,0]), rot=np.asarray([0,1,0]), rot_angle=np.asarray([(3/2)*np.pi]))
# should be [0,0,0,1]
 306:
point = point_as_quat(np.asarray([1,0,0]))
rot = rot_quat_from_axis_and_angle(axis=np.asarray([0,1,0]), angle=np.asarray([(3/2)*np.pi]))
print(conjugate_p_by_q(p=point, q=rot))
rotate_point_using_quat(p=np.asarray([1,0,0]), rot=np.asarray([0,1,0]), rot_angle=np.asarray([(3/2)*np.pi]))
# should be [0,0,0,1]
 307:
class Quat:
    def __init__(self,
                 a: float = 0,
                 b: float = 0,
                 c: float = 0,
                 d: float = 0
                )
    """
        Class that represents a quaternion. Uses numpy for calculations
    """
    self.quat = np.asarray([a, b, c, d], dtype=np.float64)
 308:
class Quat:
    def __init__(self,
                 a: float = 0,
                 b: float = 0,
                 c: float = 0,
                 d: float = 0
                ):
    """
        Class that represents a quaternion. Uses numpy for calculations
    """
    self.quat = np.asarray([a, b, c, d], dtype=np.float64)
 309:
class Quat:
    def __init__(self,
                 a: float = 0,
                 b: float = 0,
                 c: float = 0,
                 d: float = 0
                ):
        """
            Class that represents a quaternion. Uses numpy for calculations
        """
        self.quat = np.asarray([a, b, c, d], dtype=np.float64)
 310: a = Quat()
 311: a
 312:
class Quat:
    def __init__(self,
                 a: float = 0,
                 b: float = 0,
                 c: float = 0,
                 d: float = 0
                ):
        """
            Class that represents a quaternion. Uses numpy for calculations
        """
        self.quat = np.asarray([a, b, c, d], dtype=np.float64)

    def __str__(self):
        return f"Quaternion {self.quat[0]} + {self.quat[1]}i + {self.quat[2]}j + {self.quat[3]}k"
 313: a = Quat()
 314: a
 315: print(a)
 316:
class Quat:
    def __init__(self,
                 a: float = 0,
                 b: float = 0,
                 c: float = 0,
                 d: float = 0
                ):
        """
            Class that represents a quaternion. Uses numpy for calculations
        """
        self.quat = np.asarray([a, b, c, d], dtype=np.float64)


    def __str__(self):
        return f"Quaternion {self.quat[0]} + {self.quat[1]}i + {self.quat[2]}j + {self.quat[3]}k"


    def __repr__(self):
        return self.quat
 317: a = Quat()
 318: print(a)
 319: a
 320:
class Quat:
    def __init__(self,
                 a: float = 0,
                 b: float = 0,
                 c: float = 0,
                 d: float = 0
                ):
        """
            Class that represents a quaternion. Uses numpy for calculations
        """
        self.quat = np.asarray([a, b, c, d], dtype=np.float64)


    def __str__(self):
        return f"Quaternion {self.quat[0]} + {self.quat[1]}i + {self.quat[2]}j + {self.quat[3]}k"


    def __mul__(self, other):

    @staticmethod
    def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
            Calculates the Hamilton product of (possibly batched) quaternions q and p
        """
        q_single = False
        p_single = False
        if q.ndim == 1:
            q_single = True
            q = q.reshape(1, -1)
        if p.ndim == 1:
            p_single = True
            p = p.reshape(1, -1)
        if q.shape[0] == 1 and p.shape[0] > 1:
            q = np.tile(q, (p.shape[0], 1))
        if p.shape[0] == 1 and q.shape[0] > 1:
            p = np.tile(p, (q.shape[0], 1))
        assert q.shape[0] == p.shape[0], f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
        assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
        qa = q[:, 0].reshape(-1, 1)
        qb = q[:, 1].reshape(-1, 1)
        qc = q[:, 2].reshape(-1, 1)
        qd = q[:, 3].reshape(-1, 1)
        pa = p[:, 0].reshape(-1, 1)
        pb = p[:, 1].reshape(-1, 1)
        pc = p[:, 2].reshape(-1, 1)
        pd = p[:, 3].reshape(-1, 1)
        res = np.concatenate(
            [
                qa*pa - qb*pb - qc*pc - qd*pd,
                qa*pb + qb*pa + qc*pd - qd*pc,
                qa*pc - qb*pd + qc*pa + qd*pb,
                qa*pd + qb*pc - qc*pb + qd*pa
            ], axis=1, dtype=np.float64)
        if q_single and p_single:
            return res[0]
        return res
 321:
class Quat:
    def __init__(self,
                 a: float = 0,
                 b: float = 0,
                 c: float = 0,
                 d: float = 0
                ):
        """
            Class that represents a quaternion. Uses numpy for calculations
        """
        self.quat = np.asarray([a, b, c, d], dtype=np.float64)


    def __str__(self):
        return f"Quaternion {self.quat[0]} + {self.quat[1]}i + {self.quat[2]}j + {self.quat[3]}k"


    def __mul__(self, other):
        return self.hamilton_product(self.quat, other)

    @staticmethod
    def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
            Calculates the Hamilton product of (possibly batched) quaternions q and p
        """
        q_single = False
        p_single = False
        if q.ndim == 1:
            q_single = True
            q = q.reshape(1, -1)
        if p.ndim == 1:
            p_single = True
            p = p.reshape(1, -1)
        if q.shape[0] == 1 and p.shape[0] > 1:
            q = np.tile(q, (p.shape[0], 1))
        if p.shape[0] == 1 and q.shape[0] > 1:
            p = np.tile(p, (q.shape[0], 1))
        assert q.shape[0] == p.shape[0], f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
        assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
        qa = q[:, 0].reshape(-1, 1)
        qb = q[:, 1].reshape(-1, 1)
        qc = q[:, 2].reshape(-1, 1)
        qd = q[:, 3].reshape(-1, 1)
        pa = p[:, 0].reshape(-1, 1)
        pb = p[:, 1].reshape(-1, 1)
        pc = p[:, 2].reshape(-1, 1)
        pd = p[:, 3].reshape(-1, 1)
        res = np.concatenate(
            [
                qa*pa - qb*pb - qc*pc - qd*pd,
                qa*pb + qb*pa + qc*pd - qd*pc,
                qa*pc - qb*pd + qc*pa + qd*pb,
                qa*pd + qb*pc - qc*pb + qd*pa
            ], axis=1, dtype=np.float64)
        if q_single and p_single:
            return res[0]
        return res
 322: a = Quat()
 323: print(a)
 324:
a = Quat(0, 1, 2, 3)
b = Quat(1,1,0,0)
 325: print(a)
 326: a + b
 327: a * b
 328:
class Quat:
    def __init__(self,
                 a: float = 0,
                 b: float = 0,
                 c: float = 0,
                 d: float = 0
                ):
        """
            Class that represents a quaternion. Uses numpy for calculations
        """
        self.quat = np.asarray([a, b, c, d], dtype=np.float64)


    def __str__(self):
        return f"Quaternion {self.quat[0]} + {self.quat[1]}i + {self.quat[2]}j + {self.quat[3]}k"


    def __mul__(self, other):
        return self.hamilton_product(self.quat, other.quat)

    @staticmethod
    def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
            Calculates the Hamilton product of (possibly batched) quaternions q and p
        """
        q_single = False
        p_single = False
        if q.ndim == 1:
            q_single = True
            q = q.reshape(1, -1)
        if p.ndim == 1:
            p_single = True
            p = p.reshape(1, -1)
        if q.shape[0] == 1 and p.shape[0] > 1:
            q = np.tile(q, (p.shape[0], 1))
        if p.shape[0] == 1 and q.shape[0] > 1:
            p = np.tile(p, (q.shape[0], 1))
        assert q.shape[0] == p.shape[0], f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
        assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
        qa = q[:, 0].reshape(-1, 1)
        qb = q[:, 1].reshape(-1, 1)
        qc = q[:, 2].reshape(-1, 1)
        qd = q[:, 3].reshape(-1, 1)
        pa = p[:, 0].reshape(-1, 1)
        pb = p[:, 1].reshape(-1, 1)
        pc = p[:, 2].reshape(-1, 1)
        pd = p[:, 3].reshape(-1, 1)
        res = np.concatenate(
            [
                qa*pa - qb*pb - qc*pc - qd*pd,
                qa*pb + qb*pa + qc*pd - qd*pc,
                qa*pc - qb*pd + qc*pa + qd*pb,
                qa*pd + qb*pc - qc*pb + qd*pa
            ], axis=1, dtype=np.float64)
        if q_single and p_single:
            return res[0]
        return res
 329:
a = Quat(0, 1, 2, 3)
b = Quat(1,1,0,0)
 330: print(a)
 331: a * b
 332:
import numpy as np
from typing import Union
class Quat:
    def __init__(self,
                 a: Union[np.ndarray, float] = 0,
                 b: float = 0,
                 c: float = 0,
                 d: float = 0
                ):
        """
            Class that represents a quaternion. Uses numpy for calculations
        """
        if isinstance(a, np.ndarray):
            self.quat = a
        else:
            self.quat = np.asarray([a, b, c, d], dtype=np.float64)


    def __str__(self):
        return f"Quaternion {self.quat[0]} + {self.quat[1]}i + {self.quat[2]}j + {self.quat[3]}k"


    def __mul__(self, other):
        return self.hamilton_product(self.quat, other.quat)

    @staticmethod
    def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
            Calculates the Hamilton product of (possibly batched) quaternions q and p
        """
        q_single = False
        p_single = False
        if q.ndim == 1:
            q_single = True
            q = q.reshape(1, -1)
        if p.ndim == 1:
            p_single = True
            p = p.reshape(1, -1)
        if q.shape[0] == 1 and p.shape[0] > 1:
            q = np.tile(q, (p.shape[0], 1))
        if p.shape[0] == 1 and q.shape[0] > 1:
            p = np.tile(p, (q.shape[0], 1))
        assert q.shape[0] == p.shape[0], f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
        assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
        qa = q[:, 0].reshape(-1, 1)
        qb = q[:, 1].reshape(-1, 1)
        qc = q[:, 2].reshape(-1, 1)
        qd = q[:, 3].reshape(-1, 1)
        pa = p[:, 0].reshape(-1, 1)
        pb = p[:, 1].reshape(-1, 1)
        pc = p[:, 2].reshape(-1, 1)
        pd = p[:, 3].reshape(-1, 1)
        res = np.concatenate(
            [
                qa*pa - qb*pb - qc*pc - qd*pd,
                qa*pb + qb*pa + qc*pd - qd*pc,
                qa*pc - qb*pd + qc*pa + qd*pb,
                qa*pd + qb*pc - qc*pb + qd*pa
            ], axis=1, dtype=np.float64)
        if q_single and p_single:
            return res[0]
        return res
 333:
import numpy as np
from typing import Union
class Quat:
    def __init__(self,
                 a: Union[np.ndarray, float] = 0,
                 b: float = 0,
                 c: float = 0,
                 d: float = 0
                ):
        """
            Class that represents a quaternion. Uses numpy for calculations
        """
        if isinstance(a, np.ndarray):
            self.quat = a
        else:
            self.quat = np.asarray([a, b, c, d], dtype=np.float64)


    def __str__(self):
        return f"Quaternion {self.quat[0]} + {self.quat[1]}i + {self.quat[2]}j + {self.quat[3]}k"


    def __mul__(self, other):
        return Quat(self.hamilton_product(self.quat, other.quat))

    @staticmethod
    def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
            Calculates the Hamilton product of (possibly batched) quaternions q and p
        """
        q_single = False
        p_single = False
        if q.ndim == 1:
            q_single = True
            q = q.reshape(1, -1)
        if p.ndim == 1:
            p_single = True
            p = p.reshape(1, -1)
        if q.shape[0] == 1 and p.shape[0] > 1:
            q = np.tile(q, (p.shape[0], 1))
        if p.shape[0] == 1 and q.shape[0] > 1:
            p = np.tile(p, (q.shape[0], 1))
        assert q.shape[0] == p.shape[0], f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
        assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
        qa = q[:, 0].reshape(-1, 1)
        qb = q[:, 1].reshape(-1, 1)
        qc = q[:, 2].reshape(-1, 1)
        qd = q[:, 3].reshape(-1, 1)
        pa = p[:, 0].reshape(-1, 1)
        pb = p[:, 1].reshape(-1, 1)
        pc = p[:, 2].reshape(-1, 1)
        pd = p[:, 3].reshape(-1, 1)
        res = np.concatenate(
            [
                qa*pa - qb*pb - qc*pc - qd*pd,
                qa*pb + qb*pa + qc*pd - qd*pc,
                qa*pc - qb*pd + qc*pa + qd*pb,
                qa*pd + qb*pc - qc*pb + qd*pa
            ], axis=1, dtype=np.float64)
        if q_single and p_single:
            return res[0]
        return res
 334:
a = Quat(0, 1, 2, 3)
b = Quat(1,1,0,0)
 335: print(a)
 336: a * b
 337: print(a * b)
 338: a.quat
 339: print(a.quat)
 340: a.quat
 341:
import numpy as np
from typing import Union
class Quat:
    def __init__(self,
                 a: Union[np.ndarray, float] = 0,
                 b: float = 0,
                 c: float = 0,
                 d: float = 0
                ):
        """
            Class that represents a quaternion. Uses numpy for calculations
        """
        if isinstance(a, np.ndarray):
            self.quat = a
        else:
            self.quat = np.asarray([a, b, c, d], dtype=np.float64)


    def __str__(self):
        return f"Quaternion {self.quat[0]} + {self.quat[1]}i + {self.quat[2]}j + {self.quat[3]}k"


    def __repr__(self):
        return f"Quat({self.quat[0]}, {self.quat[1]}, {self.quat[2]}, {self.quat[3]}"


    def __mul__(self, other):
        return Quat(self.hamilton_product(self.quat, other.quat))

    @staticmethod
    def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
            Calculates the Hamilton product of (possibly batched) quaternions q and p
        """
        q_single = False
        p_single = False
        if q.ndim == 1:
            q_single = True
            q = q.reshape(1, -1)
        if p.ndim == 1:
            p_single = True
            p = p.reshape(1, -1)
        if q.shape[0] == 1 and p.shape[0] > 1:
            q = np.tile(q, (p.shape[0], 1))
        if p.shape[0] == 1 and q.shape[0] > 1:
            p = np.tile(p, (q.shape[0], 1))
        assert q.shape[0] == p.shape[0], f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
        assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
        qa = q[:, 0].reshape(-1, 1)
        qb = q[:, 1].reshape(-1, 1)
        qc = q[:, 2].reshape(-1, 1)
        qd = q[:, 3].reshape(-1, 1)
        pa = p[:, 0].reshape(-1, 1)
        pb = p[:, 1].reshape(-1, 1)
        pc = p[:, 2].reshape(-1, 1)
        pd = p[:, 3].reshape(-1, 1)
        res = np.concatenate(
            [
                qa*pa - qb*pb - qc*pc - qd*pd,
                qa*pb + qb*pa + qc*pd - qd*pc,
                qa*pc - qb*pd + qc*pa + qd*pb,
                qa*pd + qb*pc - qc*pb + qd*pa
            ], axis=1, dtype=np.float64)
        if q_single and p_single:
            return res[0]
        return res
 342:
a = Quat(0, 1, 2, 3)
b = Quat(1,1,0,0)
 343: a
 344:
import numpy as np
from typing import Union
class Quat:
    def __init__(self,
                 a: Union[np.ndarray, float] = 0,
                 b: float = 0,
                 c: float = 0,
                 d: float = 0
                ):
        """
            Class that represents a quaternion. Uses numpy for calculations
        """
        if isinstance(a, np.ndarray):
            self.quat = a
        else:
            self.quat = np.asarray([a, b, c, d], dtype=np.float64)


    def __str__(self):
        return f"Quaternion {self.quat[0]} + {self.quat[1]}i + {self.quat[2]}j + {self.quat[3]}k"


    def __repr__(self):
        return f"Quat({self.quat[0]}, {self.quat[1]}, {self.quat[2]}, {self.quat[3]})"


    def __mul__(self, other):
        return Quat(self.hamilton_product(self.quat, other.quat))

    @staticmethod
    def hamilton_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
            Calculates the Hamilton product of (possibly batched) quaternions q and p
        """
        q_single = False
        p_single = False
        if q.ndim == 1:
            q_single = True
            q = q.reshape(1, -1)
        if p.ndim == 1:
            p_single = True
            p = p.reshape(1, -1)
        if q.shape[0] == 1 and p.shape[0] > 1:
            q = np.tile(q, (p.shape[0], 1))
        if p.shape[0] == 1 and q.shape[0] > 1:
            p = np.tile(p, (q.shape[0], 1))
        assert q.shape[0] == p.shape[0], f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
        assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
        qa = q[:, 0].reshape(-1, 1)
        qb = q[:, 1].reshape(-1, 1)
        qc = q[:, 2].reshape(-1, 1)
        qd = q[:, 3].reshape(-1, 1)
        pa = p[:, 0].reshape(-1, 1)
        pb = p[:, 1].reshape(-1, 1)
        pc = p[:, 2].reshape(-1, 1)
        pd = p[:, 3].reshape(-1, 1)
        res = np.concatenate(
            [
                qa*pa - qb*pb - qc*pc - qd*pd,
                qa*pb + qb*pa + qc*pd - qd*pc,
                qa*pc - qb*pd + qc*pa + qd*pb,
                qa*pd + qb*pc - qc*pb + qd*pa
            ], axis=1, dtype=np.float64)
        if q_single and p_single:
            return res[0]
        return res
 345:
a = Quat(0, 1, 2, 3)
b = Quat(1,1,0,0)
 346: a.quat
 347: a
 348: print(a)
 349: print(a * b)
 350: a * b
 351: Quat
 352: a
 353: np.asarray([a])
 354:
import numpy as np
from typing import Union
class Quat:
    def __init__(self,
                 a: Union[np.ndarray, float] = 0,
                 b: float = 0,
                 c: float = 0,
                 d: float = 0
                ):
        """
            Class that represents a quaternion. Uses numpy for calculations
        """
        if isinstance(a, np.ndarray):
            self.quat = a
        else:
            self.quat = np.asarray([a, b, c, d], dtype=np.float64)


    def __str__(self):
        return f"Quaternion {self.quat[0]} + {self.quat[1]}i + {self.quat[2]}j + {self.quat[3]}k"


    def __repr__(self):
        return f"Quat({self.quat[0]}, {self.quat[1]}, {self.quat[2]}, {self.quat[3]})"


    def __mul__(self, other):
        return self.hamilton_product(self.quat, other.quat)

    @staticmethod
    def hamilton_product(q: Union[Quat, np.ndarray], p: Union[Quat, np.ndarray]) -> Quat:
        """
            Calculates the Hamilton product of (possibly batched) quaternions q and p
        """
        if isinstance(q, Quat):
            q = q.quat
        if isinstance(p, Quat):
            p = p.quat
        q_single = False
        p_single = False
        if q.ndim == 1:
            q_single = True
            q = q.reshape(1, -1)
        if p.ndim == 1:
            p_single = True
            p = p.reshape(1, -1)
        if q.shape[0] == 1 and p.shape[0] > 1:
            q = np.tile(q, (p.shape[0], 1))
        if p.shape[0] == 1 and q.shape[0] > 1:
            p = np.tile(p, (q.shape[0], 1))
        assert q.shape[0] == p.shape[0], f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
        assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
        qa = q[:, 0].reshape(-1, 1)
        qb = q[:, 1].reshape(-1, 1)
        qc = q[:, 2].reshape(-1, 1)
        qd = q[:, 3].reshape(-1, 1)
        pa = p[:, 0].reshape(-1, 1)
        pb = p[:, 1].reshape(-1, 1)
        pc = p[:, 2].reshape(-1, 1)
        pd = p[:, 3].reshape(-1, 1)
        res = np.concatenate(
            [
                qa*pa - qb*pb - qc*pc - qd*pd,
                qa*pb + qb*pa + qc*pd - qd*pc,
                qa*pc - qb*pd + qc*pa + qd*pb,
                qa*pd + qb*pc - qc*pb + qd*pa
            ], axis=1, dtype=np.float64)
        if q_single and p_single:
            return Quat(res[0])
        return np.asarray([Quat(r) for r in res])


    #@staticmethod
    #def add
 355:
a = Quat(0, 1, 2, 3)
b = Quat(1,1,0,0)
 356: np.asarray([a])
 357: a.quat
 358: a
 359: print(a)
 360: print(a * b)
 361: a * b
 362: Quat.hamilton_product?
 363: Quat.hamilton_product(q, q)
 364: Quat.hamilton_product(q, q).shape
 365: Quat.hamilton_product(q, q)[0]
 366:
class Test:
    def __init__(self):
        print("Called construtor")
    def norm(self):
        print("Called self.norm")
    @staticmethod
    def norm():
        print("Called static Test::norm")
 367:
c = Test()
c.norm()
 368:
class Test:
    def __init__(self):
        print("Called construtor")
    def norm(self):
        print("Called self.norm")
    @staticmethod
    def norm(a: Test):
        print("Called static Test::norm")
 369:
c = Test()
c.norm()
 370:
class Test:
    def __init__(self):
        print("Called construtor")
        self.norm = self._instance_norm
    def _instance_norm(self):
        print("Called class method norm")
    @staticmethod
    def norm(a: Test):
        print("Called static Test::norm")
 371:
c = Test()
c.norm()
 372:
c = Test()
c.norm()
Test.norm()
 373:
class Test:
    def __init__(self):
        print("Called construtor")
        self.norm = self._instance_norm
    def _instance_norm(self):
        print("Called class method norm")
    @staticmethod
    def norm():
        print("Called static Test::norm")
 374:
c = Test()
c.norm()
Test.norm()
 375:
a = np.asarray([1,0,0,0])
np.linalg.norm(a)
 376:
a = np.asarray([1,1,0,0])
np.linalg.norm(a)
 377:
a = Quat(0, 1, 2, 3)
b = Quat(1,1,0,0)
 378: a.quat_[0]
 379:
import numpy as np
from typing import Union
class Quat:
    def __init__(self,
                 a: Union[np.ndarray, float] = 0,
                 b: float = 0,
                 c: float = 0,
                 d: float = 0
                ):
        """
            Class that represents a quaternion in the form
            a + bi + cj + dk
            Uses numpy for calculations.
        """
        if isinstance(a, np.ndarray):
            self.quat_ = a
        else:
            self.quat_ = np.asarray([a, b, c, d], dtype=np.float64)
        self.length = self._instance_length


    def __str__(self):
        return f"Quaternion {self.quat_[0]} + {self.quat_[1]}i + {self.quat_[2]}j + {self.quat_[3]}k"


    def __repr__(self):
        return f"Quat({self.quat_[0]}, {self.quat_[1]}, {self.quat_[2]}, {self.quat_[3]})"


    def __mul__(self, other):
        return self.hamilton_product(self.quat_, other.numpy())


    def __len__(self):
        return np.linalg.norm(self.quat_)


    def numpy(self) -> np.ndarray:
        return self.quat_

    def _instance_length(self) -> float:
        return np.linalg.norm(self.quat_)
    
    def _instance_norm(self):
        

    @staticmethod
    def hamilton_product(q: Union[Quat, np.ndarray], p: Union[Quat, np.ndarray]) -> np.ndarray:
        """
            Calculates the Hamilton product of (possibly batched) quaternions q and p
        """
        if isinstance(q, Quat):
            q = q.numpy()
        if isinstance(p, Quat):
            p = p.numpy()
        q_single = False
        p_single = False
        if q.ndim == 1:
            q_single = True
            q = q.reshape(1, -1)
        if p.ndim == 1:
            p_single = True
            p = p.reshape(1, -1)
        if q.shape[0] == 1 and p.shape[0] > 1:
            q = np.tile(q, (p.shape[0], 1))
        if p.shape[0] == 1 and q.shape[0] > 1:
            p = np.tile(p, (q.shape[0], 1))
        assert q.shape[0] == p.shape[0], f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
        assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
        qa = q[:, 0].reshape(-1, 1)
        qb = q[:, 1].reshape(-1, 1)
        qc = q[:, 2].reshape(-1, 1)
        qd = q[:, 3].reshape(-1, 1)
        pa = p[:, 0].reshape(-1, 1)
        pb = p[:, 1].reshape(-1, 1)
        pc = p[:, 2].reshape(-1, 1)
        pd = p[:, 3].reshape(-1, 1)
        res = np.concatenate(
            [
                qa*pa - qb*pb - qc*pc - qd*pd,
                qa*pb + qb*pa + qc*pd - qd*pc,
                qa*pc - qb*pd + qc*pa + qd*pb,
                qa*pd + qb*pc - qc*pb + qd*pa
            ], axis=1, dtype=np.float64)
        if q_single and p_single:
            return res[0]
        return res


    @staticmethod
    def from_numpy(arr: np.ndarray) -> Union[Quat, np.ndarray]:
        if arr.ndim == 1:
            return Quat(arr)
        else:
            return np.ndarray([Quat(a) for a in arr])


    @staticmethod
    def length(q: Union[Quat, np.ndarray]) -> Union[float. np.ndarray]:
        if isinstance(q, Quat):
            q = q.numpy()
        single = False
        if q.ndim == 1:
            single = True
            q = q.reshape(1, -1)
        norm = np.linalg.norm(q, axis=1).reshape(-1, 1)
        if single:
            return norm[0]
        return norm


    @staticmethod
    def add(q: Union[Quat, np.ndarray], p: Union[Quat, np.ndarray]) -> Union[Quat, np.ndarray]:
        pass
 380:
import numpy as np
from typing import Union
class Quat:
    def __init__(self,
                 a: Union[np.ndarray, float] = 0,
                 b: float = 0,
                 c: float = 0,
                 d: float = 0
                ):
        """
            Class that represents a quaternion in the form
            a + bi + cj + dk
            Uses numpy for calculations.
        """
        if isinstance(a, np.ndarray):
            self.quat_ = a
        else:
            self.quat_ = np.asarray([a, b, c, d], dtype=np.float64)
        self.length = self._instance_length


    def __str__(self):
        return f"Quaternion {self.quat_[0]} + {self.quat_[1]}i + {self.quat_[2]}j + {self.quat_[3]}k"


    def __repr__(self):
        return f"Quat({self.quat_[0]}, {self.quat_[1]}, {self.quat_[2]}, {self.quat_[3]})"


    def __mul__(self, other):
        return self.hamilton_product(self.quat_, other.numpy())


    def __len__(self):
        return np.linalg.norm(self.quat_)


    def numpy(self) -> np.ndarray:
        return self.quat_

    def _instance_length(self) -> float:
        return np.linalg.norm(self.quat_)
    
    def _instance_norm(self):
        pass
        

    @staticmethod
    def hamilton_product(q: Union[Quat, np.ndarray], p: Union[Quat, np.ndarray]) -> np.ndarray:
        """
            Calculates the Hamilton product of (possibly batched) quaternions q and p
        """
        if isinstance(q, Quat):
            q = q.numpy()
        if isinstance(p, Quat):
            p = p.numpy()
        q_single = False
        p_single = False
        if q.ndim == 1:
            q_single = True
            q = q.reshape(1, -1)
        if p.ndim == 1:
            p_single = True
            p = p.reshape(1, -1)
        if q.shape[0] == 1 and p.shape[0] > 1:
            q = np.tile(q, (p.shape[0], 1))
        if p.shape[0] == 1 and q.shape[0] > 1:
            p = np.tile(p, (q.shape[0], 1))
        assert q.shape[0] == p.shape[0], f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
        assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
        qa = q[:, 0].reshape(-1, 1)
        qb = q[:, 1].reshape(-1, 1)
        qc = q[:, 2].reshape(-1, 1)
        qd = q[:, 3].reshape(-1, 1)
        pa = p[:, 0].reshape(-1, 1)
        pb = p[:, 1].reshape(-1, 1)
        pc = p[:, 2].reshape(-1, 1)
        pd = p[:, 3].reshape(-1, 1)
        res = np.concatenate(
            [
                qa*pa - qb*pb - qc*pc - qd*pd,
                qa*pb + qb*pa + qc*pd - qd*pc,
                qa*pc - qb*pd + qc*pa + qd*pb,
                qa*pd + qb*pc - qc*pb + qd*pa
            ], axis=1, dtype=np.float64)
        if q_single and p_single:
            return res[0]
        return res


    @staticmethod
    def from_numpy(arr: np.ndarray) -> Union[Quat, np.ndarray]:
        if arr.ndim == 1:
            return Quat(arr)
        else:
            return np.ndarray([Quat(a) for a in arr])


    @staticmethod
    def length(q: Union[Quat, np.ndarray]) -> Union[float. np.ndarray]:
        if isinstance(q, Quat):
            q = q.numpy()
        single = False
        if q.ndim == 1:
            single = True
            q = q.reshape(1, -1)
        norm = np.linalg.norm(q, axis=1).reshape(-1, 1)
        if single:
            return norm[0]
        return norm


    @staticmethod
    def add(q: Union[Quat, np.ndarray], p: Union[Quat, np.ndarray]) -> Union[Quat, np.ndarray]:
        pass
 381:
import numpy as np
from typing import Union
class Quat:
    def __init__(self,
                 a: Union[np.ndarray, float] = 0,
                 b: float = 0,
                 c: float = 0,
                 d: float = 0
                ):
        """
            Class that represents a quaternion in the form
            a + bi + cj + dk
            Uses numpy for calculations.
        """
        if isinstance(a, np.ndarray):
            self.quat_ = a
        else:
            self.quat_ = np.asarray([a, b, c, d], dtype=np.float64)
        self.length = self._instance_length


    def __str__(self):
        return f"Quaternion {self.quat_[0]} + {self.quat_[1]}i + {self.quat_[2]}j + {self.quat_[3]}k"


    def __repr__(self):
        return f"Quat({self.quat_[0]}, {self.quat_[1]}, {self.quat_[2]}, {self.quat_[3]})"


    def __mul__(self, other):
        return self.hamilton_product(self.quat_, other.numpy())


    def __len__(self):
        return np.linalg.norm(self.quat_)


    def numpy(self) -> np.ndarray:
        return self.quat_

    def _instance_length(self) -> float:
        return np.linalg.norm(self.quat_)
    
    def _instance_norm(self):
        pass
        

    @staticmethod
    def hamilton_product(q: Union[Quat, np.ndarray], p: Union[Quat, np.ndarray]) -> np.ndarray:
        """
            Calculates the Hamilton product of (possibly batched) quaternions q and p
        """
        if isinstance(q, Quat):
            q = q.numpy()
        if isinstance(p, Quat):
            p = p.numpy()
        q_single = False
        p_single = False
        if q.ndim == 1:
            q_single = True
            q = q.reshape(1, -1)
        if p.ndim == 1:
            p_single = True
            p = p.reshape(1, -1)
        if q.shape[0] == 1 and p.shape[0] > 1:
            q = np.tile(q, (p.shape[0], 1))
        if p.shape[0] == 1 and q.shape[0] > 1:
            p = np.tile(p, (q.shape[0], 1))
        assert q.shape[0] == p.shape[0], f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
        assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
        qa = q[:, 0].reshape(-1, 1)
        qb = q[:, 1].reshape(-1, 1)
        qc = q[:, 2].reshape(-1, 1)
        qd = q[:, 3].reshape(-1, 1)
        pa = p[:, 0].reshape(-1, 1)
        pb = p[:, 1].reshape(-1, 1)
        pc = p[:, 2].reshape(-1, 1)
        pd = p[:, 3].reshape(-1, 1)
        res = np.concatenate(
            [
                qa*pa - qb*pb - qc*pc - qd*pd,
                qa*pb + qb*pa + qc*pd - qd*pc,
                qa*pc - qb*pd + qc*pa + qd*pb,
                qa*pd + qb*pc - qc*pb + qd*pa
            ], axis=1, dtype=np.float64)
        if q_single and p_single:
            return res[0]
        return res


    @staticmethod
    def from_numpy(arr: np.ndarray) -> Union[Quat, np.ndarray]:
        if arr.ndim == 1:
            return Quat(arr)
        else:
            return np.ndarray([Quat(a) for a in arr])


    @staticmethod
    def length(q: Union[Quat, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(q, Quat):
            q = q.numpy()
        single = False
        if q.ndim == 1:
            single = True
            q = q.reshape(1, -1)
        norm = np.linalg.norm(q, axis=1).reshape(-1, 1)
        if single:
            return norm[0]
        return norm


    @staticmethod
    def add(q: Union[Quat, np.ndarray], p: Union[Quat, np.ndarray]) -> Union[Quat, np.ndarray]:
        pass
 382:
a = Quat(0, 1, 2, 3)
b = Quat(1,1,0,0)
 383: a.quat_[0]
 384: np.linalg.norm(a.quat_)
 385: np.asarray([1])
 386: np.asarray([1]).ndim
 387: np.asarray([1]).shape
 388: np.asarray([1]).shape[0]
 389: np.asarray([1,2,3]) * np.asarray([2])
 390: np.asarray([1,2,3]) * np.asarray([[2], [4]])
 391: np.asarray([1,2,3]) * np.asarray([2,4])
 392:
import numpy as np
from typing import Union
class Quat:
    def __init__(self,
                 a: Union[np.ndarray, float] = 0,
                 b: float = 0,
                 c: float = 0,
                 d: float = 0
                ):
        """
            Class that represents a quaternion in the form
            a + bi + cj + dk
            Uses numpy for calculations.
        """
        if isinstance(a, np.ndarray):
            self.quat_ = a
        else:
            self.quat_ = np.asarray([a, b, c, d], dtype=np.float64)
        self.length = self._instance_length
        self.norm = self._instance_norm


    def __str__(self):
        return f"Quaternion {self.quat_[0]} + {self.quat_[1]}i + {self.quat_[2]}j + {self.quat_[3]}k"


    def __repr__(self):
        return f"Quat({self.quat_[0]}, {self.quat_[1]}, {self.quat_[2]}, {self.quat_[3]})"


    def __mul__(self, other: Union[Quat, float, int, np.ndarray]):
        if isinstance(other, Quat):
            return Quat(Quat.hamilton_product(self.quat_, other.numpy()))
        if isinstance(other, np.ndarray):
            if other.ndim == 1:
                if other.shape[0] == 1:
                    other = other[0]
                else:
                    return np.asarray([Quat(self.quat_ * elem) for elem in other])
            else:
                res = Quat.hamilton_product(self.quat_, other)
                return np.asarray([Quat(r) for r in res])
        if isinstance(other, int) or isinstance(other, float):
            return Quat(self.quat_ * other)


    def __len__(self):
        return np.linalg.norm(self.quat_)


    def numpy(self) -> np.ndarray:
        return self.quat_

    def _instance_length(self) -> float:
        return np.linalg.norm(self.quat_)
    
    def _instance_norm(self) -> Quat:
        """
            Normalizes the quaternion and returns it as new Quat instance
        """
        return Quat(self.quat_ / self.length())


    def norm_(self):
        """
            Normalizes the quaternion
        """
        self.quat_ = self.quat_ / self.length()
        

    @staticmethod
    def hamilton_product(q: Union[Quat, np.ndarray], p: Union[Quat, np.ndarray]) -> np.ndarray:
        """
            Calculates the Hamilton product of (possibly batched) quaternions q and p
        """
        if isinstance(q, Quat):
            q = q.numpy()
        if isinstance(p, Quat):
            p = p.numpy()
        q_single = False
        p_single = False
        if q.ndim == 1:
            q_single = True
            q = q.reshape(1, -1)
        if p.ndim == 1:
            p_single = True
            p = p.reshape(1, -1)
        if q.shape[0] == 1 and p.shape[0] > 1:
            q = np.tile(q, (p.shape[0], 1))
        if p.shape[0] == 1 and q.shape[0] > 1:
            p = np.tile(p, (q.shape[0], 1))
        assert q.shape[0] == p.shape[0], f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
        assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
        qa = q[:, 0].reshape(-1, 1)
        qb = q[:, 1].reshape(-1, 1)
        qc = q[:, 2].reshape(-1, 1)
        qd = q[:, 3].reshape(-1, 1)
        pa = p[:, 0].reshape(-1, 1)
        pb = p[:, 1].reshape(-1, 1)
        pc = p[:, 2].reshape(-1, 1)
        pd = p[:, 3].reshape(-1, 1)
        res = np.concatenate(
            [
                qa*pa - qb*pb - qc*pc - qd*pd,
                qa*pb + qb*pa + qc*pd - qd*pc,
                qa*pc - qb*pd + qc*pa + qd*pb,
                qa*pd + qb*pc - qc*pb + qd*pa
            ], axis=1, dtype=np.float64)
        if q_single and p_single:
            return res[0]
        return res


    @staticmethod
    def from_numpy(arr: np.ndarray) -> Union[Quat, np.ndarray]:
        if arr.ndim == 1:
            return Quat(arr)
        else:
            return np.ndarray([Quat(a) for a in arr])


    @staticmethod
    def length(q: Union[Quat, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(q, Quat):
            q = q.numpy()
        single = False
        if q.ndim == 1:
            single = True
            q = q.reshape(1, -1)
        norm = np.linalg.norm(q, axis=1).reshape(-1, 1)
        if single:
            return norm[0]
        return norm


    @staticmethod
    def norm(q: Union[Quat, np.ndarray]) -> np.ndarray:
        """
            Normalizes (possibly batched) quaternion q
        """
        if isinstance(q, Quat):
            q = q.numpy()
        single_dim = False
        if q.ndim == 1:
            single_dim = True
            q = q.reshape(1, -1)
        q_norm = q / Quat.length(q)
        if single_dim:
            return q_norm[0]
        else:
            return q_norm


    @staticmethod
    def add(q: Union[Quat, np.ndarray], p: Union[Quat, np.ndarray]) -> Union[Quat, np.ndarray]:
        pass
 393:
import numpy as np
from typing import Union
class Quat:
    def __init__(self,
                 a: Union[np.ndarray, float] = 0,
                 b: float = 0,
                 c: float = 0,
                 d: float = 0
                ):
        """
            Class that represents a quaternion in the form
            a + bi + cj + dk
            Uses numpy for calculations.
        """
        if isinstance(a, np.ndarray):
            self.quat_ = a
        else:
            self.quat_ = np.asarray([a, b, c, d], dtype=np.float64)
        self.length = self._instance_length
        self.norm = self._instance_norm


    def __str__(self):
        return f"Quaternion {self.quat_[0]} + {self.quat_[1]}i + {self.quat_[2]}j + {self.quat_[3]}k"


    def __repr__(self):
        return f"Quat({self.quat_[0]}, {self.quat_[1]}, {self.quat_[2]}, {self.quat_[3]})"


    def __mul__(self, other: Union[Quat, float, int, np.ndarray]):
        if isinstance(other, Quat):
            return Quat(Quat.hamilton_product(self.quat_, other.numpy()))
        if isinstance(other, np.ndarray):
            if other.ndim == 1:
                if other.shape[0] == 1:
                    other = other[0]
                else:
                    return np.asarray([Quat(self.quat_ * elem) for elem in other])
            else:
                res = Quat.hamilton_product(self.quat_, other)
                return np.asarray([Quat(r) for r in res])
        if isinstance(other, int) or isinstance(other, float):
            return Quat(self.quat_ * other)


    def __len__(self):
        return np.linalg.norm(self.quat_)


    def numpy(self) -> np.ndarray:
        return self.quat_

    def _instance_length(self) -> float:
        return np.linalg.norm(self.quat_)
    
    def _instance_norm(self) -> Quat:
        """
            Normalizes the quaternion and returns it as new Quat instance
        """
        return Quat(self.quat_ / self.length())


    def norm_(self):
        """
            Normalizes the quaternion
        """
        self.quat_ = self.quat_ / self.length()
        

    @staticmethod
    def hamilton_product(q: Union[Quat, np.ndarray], p: Union[Quat, np.ndarray]) -> np.ndarray:
        """
            Calculates the Hamilton product of (possibly batched) quaternions q and p
        """
        if isinstance(q, Quat):
            q = q.numpy()
        if isinstance(p, Quat):
            p = p.numpy()
        q_single = False
        p_single = False
        if q.ndim == 1:
            q_single = True
            q = q.reshape(1, -1)
        if p.ndim == 1:
            p_single = True
            p = p.reshape(1, -1)
        if q.shape[0] == 1 and p.shape[0] > 1:
            q = np.tile(q, (p.shape[0], 1))
        if p.shape[0] == 1 and q.shape[0] > 1:
            p = np.tile(p, (q.shape[0], 1))
        assert q.shape[0] == p.shape[0], f"Both quaternions have to have the same batch size, got {q.shape[0]} and {p.shape[0]}"
        assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"
        qa = q[:, 0].reshape(-1, 1)
        qb = q[:, 1].reshape(-1, 1)
        qc = q[:, 2].reshape(-1, 1)
        qd = q[:, 3].reshape(-1, 1)
        pa = p[:, 0].reshape(-1, 1)
        pb = p[:, 1].reshape(-1, 1)
        pc = p[:, 2].reshape(-1, 1)
        pd = p[:, 3].reshape(-1, 1)
        res = np.concatenate(
            [
                qa*pa - qb*pb - qc*pc - qd*pd,
                qa*pb + qb*pa + qc*pd - qd*pc,
                qa*pc - qb*pd + qc*pa + qd*pb,
                qa*pd + qb*pc - qc*pb + qd*pa
            ], axis=1, dtype=np.float64)
        if q_single and p_single:
            return res[0]
        return res


    @staticmethod
    def from_numpy(arr: np.ndarray) -> Union[Quat, np.ndarray]:
        if arr.ndim == 1:
            return Quat(arr)
        else:
            return np.ndarray([Quat(a) for a in arr])


    @staticmethod
    def length(q: Union[Quat, np.ndarray]) -> Union[float, np.ndarray]:
        if isinstance(q, Quat):
            q = q.numpy()
        single = False
        if q.ndim == 1:
            single = True
            q = q.reshape(1, -1)
        norm = np.linalg.norm(q, axis=1).reshape(-1, 1)
        if single:
            return norm[0]
        return norm


    @staticmethod
    def norm(q: Union[Quat, np.ndarray]) -> np.ndarray:
        """
            Normalizes (possibly batched) quaternion q
        """
        if isinstance(q, Quat):
            q = q.numpy()
        single_dim = False
        if q.ndim == 1:
            single_dim = True
            q = q.reshape(1, -1)
        q_norm = q / Quat.length(q)
        if single_dim:
            return q_norm[0]
        else:
            return q_norm


    @staticmethod
    def add(q: Union[Quat, np.ndarray], p: Union[Quat, np.ndarray]) -> Union[Quat, np.ndarray]:
        pass
 394: np.asarray([1,2,3]) + np.asarray([1,2,3])
 395: np.asarray([1,2,3]) + np.asarray([[1,2,3], [4,5,6]])
 396: np.asarray([[1,2,3], [1,2,3]]) + np.asarray([[1,2,3], [4,5,6]])
 397: 5
 398: ~5
 399: ~1
 400: np.asarray([[1,2,3], [1,2,3]])**2
 401: np.asarray([[1,2,3], [1,2,3]]) / np.linalg.norm(np.asarray([[1,2,3], [1,2,3]]), axis=1)**2
 402: np.linalg.norm(np.asarray([[1,2,3], [1,2,3]]), axis=1)
 403: np.asarray([[1,2,3], [1,2,3]]) / np.linalg.norm(np.asarray([[1,2,3], [1,2,3]]), axis=1)**2.reshape(-1, 1)
 404: np.asarray([[1,2,3], [1,2,3]]) / (np.linalg.norm(np.asarray([[1,2,3], [1,2,3]]), axis=1)**2).reshape(-1, 1)
 405: np.linalg.norm?
 406: %history
 407: %history -g -f test.py
