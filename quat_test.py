import unittest
import numpy as np
from quat import Quat
import math

class TestQuat(unittest.TestCase):

    def test_init_from_np_arr(self):
        a = np.asarray([1,0,0,0])
        b = Quat(a)
        self.assertTrue((b.numpy() == a).all())


    def test_init_from_values(self):
        a = np.asarray([1,0,0,0])
        b = Quat(a[0], a[1], a[2], a[3])
        self.assertTrue((b.numpy() == a).all())


    def test_from_point(self):
        with self.assertRaises(TypeError):
            Quat.from_point(1)
        with self.assertRaises(TypeError):
            Quat.from_point("test")
        with self.assertRaises(ValueError):
            Quat.from_point(np.random.rand(2))
        with self.assertRaises(ValueError):
            Quat.from_point(np.random.rand(5, 4))
        p = np.random.rand(3) - 0.5
        q = Quat.from_point(p)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.vector(), p).all())
        self.assertTrue(q.is_pure())
        ps = np.random.rand(10, 3) - 0.5
        qs = Quat.from_point(ps)
        self.assertIsInstance(qs, np.ndarray)
        for i, q in enumerate(qs):
            self.assertIsInstance(q, Quat)
            self.assertTrue(np.isclose(q.vector(), ps[i]).all())
            self.assertTrue(q.is_pure())


    # This tests multiple components at once
    def test_rotation(self):
        def calc_rotation(axis, angle, point):
            q = Quat.from_axis_and_angle(axis, angle)
            p = Quat.from_point(point)
            p_rot = Quat.rotate_point(p, q)
            p_hat = p_rot.vector()
            return p_hat
        axis = np.asarray([0,0,1])
        angle = np.pi / 2
        point = np.asarray([2,0,0])
        p_hat = calc_rotation(axis = axis,
                              angle = angle,
                              point = point
                              )
        self.assertTrue(np.isclose(p_hat, np.asarray([0,2,0])).all())
        self.assertTrue(np.isclose(p_hat, Quat.rotate(point, axis, angle)).all())

        angle = np.pi
        p_hat = calc_rotation(axis = axis,
                              angle = angle,
                              point = point
                              )
        self.assertTrue(np.isclose(p_hat, np.asarray([-2,0,0])).all())
        self.assertTrue(np.isclose(p_hat, Quat.rotate(point, axis, angle)).all())

        angle = 3/2 * np.pi
        p_hat = calc_rotation(axis = axis,
                              angle = angle,
                              point = point
                              )
        self.assertTrue(np.isclose(p_hat, np.asarray([0,-2,0])).all())
        self.assertTrue(np.isclose(p_hat, Quat.rotate(point, axis, angle)).all())

        angle = 2 * np.pi
        p_hat = calc_rotation(axis = axis,
                              angle = angle,
                              point = point
                              )
        self.assertTrue(np.isclose(p_hat, np.asarray([2,0,0])).all())
        self.assertTrue(np.isclose(p_hat, Quat.rotate(point, axis, angle)).all())

        axis = np.asarray([0,1,0])
        angle = np.pi/2
        p_hat = calc_rotation(axis = axis,
                              angle = angle,
                              point = point
                              )
        self.assertTrue(np.isclose(p_hat, np.asarray([0,0,-2])).all())
        self.assertTrue(np.isclose(p_hat, Quat.rotate(point, axis, angle)).all())

        axis = np.asarray([1,0,0])
        point = np.asarray([0,2,0])
        p_hat = calc_rotation(axis = axis,
                              angle = angle,
                              point = point
                              )
        self.assertTrue(np.isclose(p_hat, np.asarray([0,0,2])).all())
        self.assertTrue(np.isclose(p_hat, Quat.rotate(point, axis, angle)).all())

        # rotate multiple points by same axis and angle
        points = np.asarray([[1,0,0], [2,0,0], [0,2,0], [0,0,3]])
        axis = np.asarray([0,0,3])
        angle = np.pi / 2
        p_hat = Quat.rotate(points, axis, angle)
        self.assertIsInstance(p_hat, np.ndarray)
        self.assertTrue(np.isclose(p_hat[0], np.asarray([0,1,0])).all())
        self.assertTrue(np.isclose(p_hat[1], np.asarray([0,2,0])).all())
        self.assertTrue(np.isclose(p_hat[2], np.asarray([-2,0,0])).all())
        self.assertTrue(np.isclose(p_hat[3], np.asarray([0,0,3])).all())

        # rotate same point by same axis but multiple angles
        point = np.asarray([1,0,0])
        axis = np.asarray([0,0,1])
        angles = np.asarray([np.pi / 2, np.pi, 3/2 * np.pi, 2 * np.pi]).reshape(-1, 1)
        p_hat = Quat.rotate(point, axis, angles)
        self.assertIsInstance(p_hat, np.ndarray)
        self.assertTrue(np.isclose(p_hat[0], np.asarray([0,1,0])).all())
        self.assertTrue(np.isclose(p_hat[1], np.asarray([-1,0,0])).all())
        self.assertTrue(np.isclose(p_hat[2], np.asarray([0,-1,0])).all())
        self.assertTrue(np.isclose(p_hat[3], np.asarray([1,0,0])).all())

        # rotate same point by same angle but around different axes
        point = np.asarray([1,0,0])
        angle = np.pi
        axes = np.asarray([[1,0,0], [0,1,0], [0,0,1]])
        p_hat = Quat.rotate(point, axes, angle)
        self.assertIsInstance(p_hat, np.ndarray)
        self.assertTrue(np.isclose(p_hat[0], np.asarray([1,0,0])).all())
        self.assertTrue(np.isclose(p_hat[1], np.asarray([-1,0,0])).all())
        self.assertTrue(np.isclose(p_hat[2], np.asarray([-1,0,0])).all())

        # rotate 3 different points by 3 different angles around 3 different axes
        points = np.asarray([[1,0,0], [0,1,0], [0,0,1]])
        angles = np.asarray([np.pi, np.pi / 2, 3/2 * np.pi]).reshape(-1, 1)
        axes = np.asarray([[0,0,1], [1,0,0], [0,1,0]])
        p_hat = Quat.rotate(points, axes, angles)
        self.assertIsInstance(p_hat, np.ndarray)
        self.assertTrue(np.isclose(p_hat[0], np.asarray([-1,0,0])).all())
        self.assertTrue(np.isclose(p_hat[1], np.asarray([0,0,1])).all())
        self.assertTrue(np.isclose(p_hat[2], np.asarray([-1,0,0])).all())

        # rotate 3 different points around 3 different axes by the same angle
        angle = np.pi / 2
        points = np.asarray([[1,0,0], [0,1,0], [0,0,1]])
        axes = np.asarray([[0,0,1], [1,0,0], [0,1,0]])
        p_hat = Quat.rotate(points, axes, angle)
        self.assertIsInstance(p_hat, np.ndarray)
        self.assertTrue(np.isclose(p_hat[0], np.asarray([0,1,0])).all())
        self.assertTrue(np.isclose(p_hat[1], np.asarray([0,0,1])).all())
        self.assertTrue(np.isclose(p_hat[2], np.asarray([1,0,0])).all())

        # rotate the same point around 3 different axes by 3 different angles
        point = np.asarray([1,0,0])
        axes = np.asarray([[1,0,0], [0,1,0], [0,0,1]])
        angles = np.asarray([np.pi, np.pi/2, 3/2*np.pi]).reshape(-1, 1)
        p_hat = Quat.rotate(point, axes, angles)
        self.assertIsInstance(p_hat, np.ndarray)
        self.assertTrue(np.isclose(p_hat[0], np.asarray([1,0,0])).all())
        self.assertTrue(np.isclose(p_hat[1], np.asarray([0,0,-1])).all())
        self.assertTrue(np.isclose(p_hat[2], np.asarray([0,-1,0])).all())

        # rotate 3 different points by 3 different angles around the same axis
        axis = np.asarray([0,0,1])
        points = np.asarray([[1,0,0], [2,0,0], [3,0,0]])
        angles = np.asarray([np.pi, 3/2 * np.pi, 5/2 * np.pi]).reshape(-1, 1)
        p_hat = Quat.rotate(points, axis, angles)
        self.assertIsInstance(p_hat, np.ndarray)
        self.assertTrue(np.isclose(p_hat[0], np.asarray([-1,0,0])).all())
        self.assertTrue(np.isclose(p_hat[1], np.asarray([0,-2,0])).all())
        self.assertTrue(np.isclose(p_hat[2], np.asarray([0,3,0])).all())

        # make sure angles have to be formatted correctly
        angles = angles.reshape(-1)
        with self.assertRaises(ValueError):
            Quat.rotate(points, axis, angles)
        # make sure that batch dimensions of all arrays are compatible
        axes = np.asarray([[1,0,0], [0,1,0]])
        points = np.asarray([[0,0,1], [0,0,1], [0,0,1]])
        angles = np.asarray([np.pi, np.pi, np.pi, np.pi]).reshape(-1, 1)
        with self.assertRaises(AssertionError):
            Quat.rotate(points, axes, angles)


    def test_repr(self):
        a = np.random.rand(4) - 0.5
        b = Quat(a)
        self.assertEqual(b.__repr__(), f"Quat({a[0]}, {a[1]}, {a[2]}, {a[3]})")


    def test_str(self):
        a = np.random.rand(4) - 0.5
        b = Quat(a)
        self.assertEqual(str(b), f"Quaternion {a[0]} + {a[1]}i + {a[2]}j + {a[3]}k")


    def test_neg(self):
        a = np.random.rand(4) - 0.5
        q = Quat(a)
        q_neg = -q
        self.assertIsInstance(q_neg, Quat)
        self.assertTrue(np.isclose(q_neg.numpy(), a * -1).all())


    #################################
    #   Start Testing of Addition   #
    #################################

    def test_static_add_single_numpy(self):
        a = np.random.rand(4) - 0.5
        b = np.random.rand(4) - 0.5
        c = Quat.add(a, b)
        self.assertIsInstance(c, np.ndarray, "Addition is not a numpy array")
        self.assertTrue((c == (a+b)).all(), "Addition is not correct")


    def test_static_add_single_quat(self):
        a = np.random.rand(4) - 0.5
        b = np.random.rand(4) - 0.5
        c = Quat(a)
        d = Quat(b)
        e = Quat.add(c, d)
        self.assertIsInstance(e, np.ndarray)
        self.assertTrue((e == (a+b)).all())


    def test_static_add_multiple_numpy(self):        
        for i in range(10):
            a_num = np.random.randint(1, 100)
            a = np.random.rand(a_num, 4) - 0.5
            if i % 2 == 0:
                b = np.random.rand(1, 4) - 0.5
            else:
                b = np.random.rand(4) - 0.5
            c = Quat.add(a, b)
            self.assertIsInstance(c, np.ndarray)
            self.assertTrue((c == (a+b)).all())
        for i in range(10):
            b_num = np.random.randint(1, 100)
            if i % 2 == 0:
                a = np.random.rand(1, 4) - 0.5
            else:
                a = np.random.rand(4) - 0.5
            b = np.random.rand(b_num, 4) - 0.5
            c = Quat.add(a, b)
            self.assertIsInstance(c, np.ndarray)
            self.assertTrue((c == (a+b)).all())


    def test_static_add_single_mixed(self):
        a = np.random.rand(4) - 0.5
        b = np.random.rand(4) - 0.5
        c = Quat(a)
        d = Quat.add(c, b)
        self.assertIsInstance(d, np.ndarray)
        self.assertTrue((d == (a+b)).all())
        e = Quat.add(b, c)
        self.assertIsInstance(d, np.ndarray)
        self.assertTrue((e == (a+b)).all())


    def test_static_add_multiple_mixed(self):
        a = np.random.rand(4) - 0.5
        b = np.random.rand(3, 4) - 0.5
        c = Quat(a)
        d = Quat.add(c, b)
        e = Quat.add(b, c)
        self.assertIsInstance(d, np.ndarray)
        self.assertTrue((d == (a+b)).all())
        self.assertIsInstance(e, np.ndarray)
        self.assertTrue((e == (a+b)).all())


    def test_member_add_single(self):
        a = np.random.rand(4) - 0.5
        b = np.random.rand(4) - 0.5
        c = np.random.rand(4) - 0.5
        q = Quat(a)
        p = Quat(b)
        d = q.add(c)
        self.assertIsInstance(d, Quat)
        self.assertTrue((d.numpy() == a+c).all())
        e = q.add(p)
        self.assertIsInstance(e, Quat)
        self.assertTrue((e.numpy() == a+b).all())


    def test_member_add_multiple(self):
        a = np.random.rand(4) - 0.5
        q = Quat(a)

        b = np.random.rand(10, 4) - 0.5
        c = q.add(b)
        self.assertIsInstance(c, np.ndarray)
        for idx, x in enumerate(c):
            self.assertIsInstance(x, Quat)
            self.assertTrue((x.numpy() == a+b[idx]).all())


    def test_member_add_inplace(self):
        a = np.random.rand(4) - 0.5
        q = Quat(a)
        b = np.random.rand(4) - 0.5
        p = Quat(b)
        q.add_(b)
        self.assertIsInstance(q, Quat)
        self.assertTrue((q.numpy() == a+b).all())
        q = Quat(a)
        q.add_(p)
        self.assertIsInstance(q, Quat)
        self.assertTrue((q.numpy() == a+b).all())
        c = np.random.rand(1,4) - 0.5
        q.add_(c)
        self.assertRaises(ValueError, q.add_, np.random.rand(2, 4) - 0.5)


    def test_operator_add_single(self):
        a = np.random.rand(4) - 0.5
        b = np.random.rand(4) - 0.5
        q = Quat(a)
        p = Quat(b)
        c = q + p
        d = p + q
        self.assertIsInstance(c, Quat)
        self.assertIsInstance(d, Quat)
        self.assertTrue((c.numpy() == a+b).all())
        self.assertTrue((d.numpy() == b+a).all())
        self.assertTrue((c.numpy() == d.numpy()).all())


    def test_operator_add_multiple(self):
        a = np.random.rand(4) - 0.5
        q = Quat(a)
        b = np.random.rand(10,4) - 0.5
        c = q + b
        self.assertIsInstance(c, np.ndarray)
        for idx, x in enumerate(c):
            self.assertIsInstance(x, Quat)
            self.assertTrue((x.numpy() == a+b[idx]).all())
        d = b + q
        self.assertIsInstance(d, np.ndarray)
        for idx, x in enumerate(d):
            self.assertIsInstance(x, Quat)
            self.assertTrue((x.numpy() == b[idx]+a).all())


    def test_operator_add_inplace(self):
        a = np.random.rand(4) - 0.5
        q = Quat(a)
        b = np.random.rand(4) - 0.5
        p = Quat(b)
        q += p
        self.assertIsInstance(q, Quat)
        self.assertTrue((q.numpy() == a+b).all())

    ###############################
    #   End Testing of Addition   #
    ###############################


    ####################################
    #   Start Testing of Subtraction   #
    ####################################

    def test_static_sub_single_numpy(self):
        a = np.random.rand(4) - 0.5
        b = np.random.rand(4) - 0.5
        c = Quat.sub(a, b)
        self.assertIsInstance(c, np.ndarray, "Subtraction is not a numpy array")
        self.assertTrue((c == (a-b)).all(), "Subtraction is not correct")
        d = Quat.sub(b, a)
        self.assertIsInstance(d, np.ndarray)
        self.assertTrue((d == b-a).all())


    def test_static_sub_single_quat(self):
        a = np.random.rand(4) - 0.5
        b = np.random.rand(4) - 0.5
        c = Quat(a)
        d = Quat(b)
        e = Quat.sub(c, d)
        self.assertIsInstance(e, np.ndarray)
        self.assertTrue((e == (a-b)).all())
        f = Quat.sub(d, c)
        self.assertIsInstance(f, np.ndarray)
        self.assertTrue((f == b-a).all())


    def test_static_sub_multiple_numpy(self):        
        for i in range(10):
            a_num = np.random.randint(1, 100)
            a = np.random.rand(a_num, 4) - 0.5
            if i % 2 == 0:
                b = np.random.rand(1, 4) - 0.5
            else:
                b = np.random.rand(4) - 0.5
            c = Quat.sub(a, b)
            self.assertIsInstance(c, np.ndarray)
            self.assertTrue((c == (a-b)).all())
            d = Quat.sub(b, a)
            self.assertIsInstance(d, np.ndarray)
            self.assertTrue((d == b-a).all())
        for i in range(10):
            b_num = np.random.randint(1, 100)
            if i % 2 == 0:
                a = np.random.rand(1, 4) - 0.5
            else:
                a = np.random.rand(4) - 0.5
            b = np.random.rand(b_num, 4) - 0.5
            c = Quat.sub(a, b)
            self.assertIsInstance(c, np.ndarray)
            self.assertTrue((c == (a-b)).all())
            d = Quat.sub(b, a)
            self.assertIsInstance(d, np.ndarray)
            self.assertTrue((d == b-a).all())


    def test_static_sub_single_mixed(self):
        a = np.random.rand(4) - 0.5
        b = np.random.rand(4) - 0.5
        c = Quat(a)
        d = Quat.sub(c, b)
        self.assertIsInstance(d, np.ndarray)
        self.assertTrue((d == (a-b)).all())
        e = Quat.sub(b, c)
        self.assertIsInstance(d, np.ndarray)
        self.assertTrue((e == (b-a)).all())


    def test_static_sub_multiple_mixed(self):
        a = np.random.rand(4) - 0.5
        b = np.random.rand(3, 4) - 0.5
        c = Quat(a)
        d = Quat.sub(c, b)
        e = Quat.sub(b, c)
        self.assertIsInstance(d, np.ndarray)
        self.assertTrue((d == (a-b)).all())
        self.assertIsInstance(e, np.ndarray)
        self.assertTrue((e == (b-a)).all())


    def test_member_sub_single(self):
        a = np.random.rand(4) - 0.5
        b = np.random.rand(4) - 0.5
        c = np.random.rand(4) - 0.5
        q = Quat(a)
        p = Quat(b)
        d = q.sub(c)
        self.assertIsInstance(d, Quat)
        self.assertTrue((d.numpy() == a-c).all())
        e = q.sub(p)
        self.assertIsInstance(e, Quat)
        self.assertTrue((e.numpy() == a-b).all())


    def test_member_sub_multiple(self):
        a = np.random.rand(4) - 0.5
        q = Quat(a)

        b = np.random.rand(10, 4) - 0.5
        c = q.sub(b)
        self.assertIsInstance(c, np.ndarray)
        for idx, x in enumerate(c):
            self.assertIsInstance(x, Quat)
            self.assertTrue((x.numpy() == a-b[idx]).all())


    def test_member_sub_inplace(self):
        a = np.random.rand(4) - 0.5
        q = Quat(a)
        b = np.random.rand(4) - 0.5
        p = Quat(b)
        q.sub_(b)
        self.assertIsInstance(q, Quat)
        self.assertTrue((q.numpy() == a-b).all())
        q = Quat(a)
        q.sub_(p)
        self.assertIsInstance(q, Quat)
        self.assertTrue((q.numpy() == a-b).all())
        c = np.random.rand(1,4) - 0.5
        q.sub_(c)
        self.assertRaises(ValueError, q.sub_, np.random.rand(2, 4) - 0.5)


    def test_operator_sub_single(self):
        a = np.random.rand(4) - 0.5
        b = np.random.rand(4) - 0.5
        q = Quat(a)
        p = Quat(b)
        c = q - p
        d = p - q
        self.assertIsInstance(c, Quat)
        self.assertIsInstance(d, Quat)
        self.assertTrue((c.numpy() == a-b).all())
        self.assertTrue((d.numpy() == b-a).all())


    def test_operator_sub_multiple(self):
        a = np.random.rand(4) - 0.5
        q = Quat(a)
        b = np.random.rand(10,4) - 0.5
        c = q - b
        self.assertIsInstance(c, np.ndarray)
        for idx, x in enumerate(c):
            self.assertIsInstance(x, Quat)
            self.assertTrue((x.numpy() == a-b[idx]).all())
        d = b - q
        self.assertIsInstance(d, np.ndarray)
        for idx, x in enumerate(d):
            self.assertIsInstance(x, Quat)
            self.assertTrue((x.numpy() == b[idx]-a).all())


    def test_operator_sub_inplace(self):
        a = np.random.rand(4) - 0.5
        q = Quat(a)
        b = np.random.rand(4) - 0.5
        p = Quat(b)
        q -= p
        self.assertIsInstance(q, Quat)
        self.assertTrue((q.numpy() == a-b).all())
    
    ##################################
    #   End Testing of Subtraction   #
    ##################################


    ########################################
    #   Start Testing of Scalar Multiply   #
    ########################################

    def test_static_scalar_multiply(self):
        q_single = Quat(np.random.rand(4) - 0.5)
        q_multi = np.random.rand(10, 4) - 0.5
        s_single_int = np.random.randint(-10, 10)
        s_single_float = np.random.rand(1)[0] - 0.5

        # Single / multi quaternion times single scalar as int or float
        res = Quat.scalar_multiply(q_single, s_single_int)
        self.assertTrue(np.isclose(res, q_single.numpy() * s_single_int).all())
        self.assertTrue(res.ndim == 1)
        res = Quat.scalar_multiply(q_multi, s_single_int)
        self.assertTrue(np.isclose(res, q_multi*s_single_int).all())
        self.assertTrue(res.ndim == 2)
        res = Quat.scalar_multiply(q_single, s_single_float)
        self.assertTrue(np.isclose(res, q_single.numpy() * s_single_float).all())
        self.assertTrue(res.ndim == 1)
        res = Quat.scalar_multiply(q_multi, s_single_float)
        self.assertTrue(np.isclose(res, q_multi*s_single_float).all())
        self.assertTrue(res.ndim == 2)

        # Single / multi quaternion times single / multi scalar as np array
        s_single_arr_onedim = np.random.rand(1) - 0.5
        s_single_arr_multidim = np.random.rand(1, 1) - 0.5
        s_multi_arr = np.random.rand(10, 1) - 0.5
        res = Quat.scalar_multiply(q_single, s_single_arr_onedim)
        self.assertTrue(np.isclose(res, q_single.numpy() * s_single_arr_onedim).all())
        self.assertTrue(res.ndim == 1)
        res = Quat.scalar_multiply(q_multi, s_single_arr_onedim)
        self.assertTrue(np.isclose(res, q_multi * s_single_arr_onedim).all())
        self.assertTrue(res.ndim == 2)
        res = Quat.scalar_multiply(q_single, s_single_arr_multidim)
        self.assertTrue(np.isclose(res, q_single.numpy() * s_single_arr_multidim).all())
        self.assertTrue(res.ndim == 2)
        res = Quat.scalar_multiply(q_multi, s_single_arr_multidim)
        self.assertTrue(np.isclose(res, q_multi * s_single_arr_multidim).all())
        self.assertTrue(res.ndim == 2)
        res = Quat.scalar_multiply(q_single, s_multi_arr)
        self.assertTrue(np.isclose(res, q_single.numpy() * s_multi_arr).all())
        self.assertTrue(res.ndim == 2)
        res = Quat.scalar_multiply(q_multi, s_multi_arr)
        self.assertTrue(np.isclose(res, q_multi * s_multi_arr).all())
        self.assertTrue(res.ndim == 2)
        q_multi_wrong_dim = np.random.rand(15, 4) - 0.5
        self.assertRaises(ValueError, Quat.scalar_multiply, q_multi_wrong_dim, s_multi_arr)
        self.assertRaises(ValueError, Quat.scalar_multiply, np.asarray(5), 1)
        self.assertRaises(ValueError, Quat.scalar_multiply, np.random.rand(2, 5), 1)
        self.assertRaises(ValueError, Quat.scalar_multiply, q_single, np.random.rand(4))


    def test_operator_multiply(self):
        q = Quat(np.random.rand(4) - 0.5)
        s_int = np.random.randint(-10, 10)
        s_float = np.random.rand(1)[0] - 0.5
        s_arr_onedim_single = np.random.rand(1) - 0.5
        s_arr_twodim_single = np.random.rand(1, 1) - 0.5
        s_arr_twodim_multi = np.random.rand(10, 1) - 0.5
        p = Quat(np.random.rand(4) - 0.5)
        o = np.random.rand(4) - 0.5
        n = np.random.rand(10, 4) - 0.5

        # Quat * scalars
        res = q * s_int
        self.assertIsInstance(res, Quat)
        self.assertTrue(np.isclose(res.numpy(), q.numpy() * s_int).all())
        res = q * s_float
        self.assertIsInstance(res, Quat)
        self.assertTrue(np.isclose(res.numpy(), q.numpy() * s_float).all())
        res = q * s_arr_onedim_single
        self.assertIsInstance(res, Quat)
        self.assertTrue(np.isclose(res.numpy(), q.numpy() * s_arr_onedim_single).all())
        res = q * s_arr_twodim_single
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(len(res) == s_arr_twodim_single.shape[0])
        for i, x in enumerate(res):
            self.assertIsInstance(x, Quat)
            self.assertTrue(np.isclose(x.numpy(), q.numpy() * s_arr_twodim_single[i]).all())
        res = q * s_arr_twodim_multi
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(len(res) == s_arr_twodim_multi.shape[0])
        for i, x in enumerate(res):
            self.assertIsInstance(x, Quat)
            self.assertTrue(np.isclose(x.numpy(), q.numpy() * s_arr_twodim_multi[i]).all())

        # Quat * Quat should not be supported by scalar multiplication *
        with self.assertRaises(TypeError):
            q * p
        with self.assertRaises(ValueError):
            q * o
        with self.assertRaises(ValueError):
            q * n
        with self.assertRaises(ValueError):
            q * (np.random.rand(5) - 0.5)

        # res = q * p
        # self.assertIsInstance(res, Quat)
        # self.assertTrue(np.isclose(res.numpy(), Quat.hamilton_product(q, p)).all())
        # res = q * o
        # self.assertIsInstance(res, Quat)
        # self.assertTrue(np.isclose(res.numpy(), Quat.hamilton_product(q, o)).all())
        # res = q * n
        # self.assertIsInstance(res, np.ndarray)
        # self.assertTrue(len(res) == n.shape[0])
        # for i, x in enumerate(res):
        #     self.assertIsInstance(x, Quat)
        #     self.assertTrue(np.isclose(x.numpy(), Quat.hamilton_product(q.numpy(), n[i])).all())

        # Check errors are raised for invalid input
        with self.assertRaises(ValueError):
            q * (np.random.rand(2) - 0.5)
        with self.assertRaises(ValueError):
            q * (np.random.rand(6) - 0.5)
        with self.assertRaises(ValueError):
            q * (np.random.rand(10, 2) - 0.5)


    def test_operator_reverse_multiply(self):
        q = Quat(np.random.rand(4) - 0.5)
        s_int = np.random.randint(-10, 10)
        s_float = np.random.rand(1)[0] - 0.5
        s_arr_onedim_single = np.random.rand(1) - 0.5
        s_arr_twodim_single = np.random.rand(1, 1) - 0.5
        s_arr_twodim_multi = np.random.rand(10, 1) - 0.5
        o = np.random.rand(4) - 0.5
        n = np.random.rand(10, 4) - 0.5

        # Quat * scalars
        res = s_int * q
        self.assertIsInstance(res, Quat)
        self.assertTrue(np.isclose(res.numpy(), q.numpy() * s_int).all())
        res = s_float * q
        self.assertIsInstance(res, Quat)
        self.assertTrue(np.isclose(res.numpy(), q.numpy() * s_float).all())
        res = s_arr_onedim_single * q
        self.assertIsInstance(res, Quat)
        self.assertTrue(np.isclose(res.numpy(), q.numpy() * s_arr_onedim_single).all())
        res = s_arr_twodim_single * q
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(len(res) == s_arr_twodim_single.shape[0])
        for i, x in enumerate(res):
            self.assertIsInstance(x, Quat)
            self.assertTrue(np.isclose(x.numpy(), q.numpy() * s_arr_twodim_single[i]).all())
        res = s_arr_twodim_multi * q
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(len(res) == s_arr_twodim_multi.shape[0])
        for i, x in enumerate(res):
            self.assertIsInstance(x, Quat)
            self.assertTrue(np.isclose(x.numpy(), q.numpy() * s_arr_twodim_multi[i]).all())

        # Quat * Quat should not be supported for scalar multiply *
        with self.assertRaises(ValueError):
            o * q
        with self.assertRaises(ValueError):
            n * q

        # res = o * q
        # self.assertIsInstance(res, Quat)
        # self.assertTrue(np.isclose(res.numpy(), Quat.hamilton_product(o, q)).all())
        # res = n * q
        # self.assertIsInstance(res, np.ndarray)
        # self.assertTrue(len(res) == n.shape[0])
        # for i, x in enumerate(res):
        #     self.assertIsInstance(x, Quat)
        #     self.assertTrue(np.isclose(x.numpy(), Quat.hamilton_product(n[i], q.numpy())).all())

        # Check errors are raised for invalid input
        with self.assertRaises(ValueError):
            (np.random.rand(2) - 0.5) * q
        with self.assertRaises(ValueError):
            (np.random.rand(6) - 0.5) * q
        with self.assertRaises(ValueError):
            (np.random.rand(10, 2) - 0.5) * q


    def test_operator_inplace_multiply(self):
        a = np.random.rand(4) - 0.5
        q = Quat(a)
        s_int = np.random.randint(-10, 10)
        s_float = np.random.rand(1)[0] - 0.5
        s_arr_onedim_single = np.random.rand(1) - 0.5
        s_arr_twodim_single = np.random.rand(1, 1) - 0.5
        p = Quat(np.random.rand(4) - 0.5)
        o = np.random.rand(4) - 0.5
        n = np.random.rand(1, 4) - 0.5

        # Quat *= scalars
        q *= s_int
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), a * s_int).all())
        q = Quat(a)
        q *= s_float
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), a * s_float).all())
        q = Quat(a)
        q *= s_arr_onedim_single
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), a * s_arr_onedim_single).all())
        q = Quat(a)
        q *= s_arr_twodim_single
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), a * s_arr_twodim_single).all())

        # Quat *= Quat should not be supported by scalar multiply *
        q = Quat(a)
        with self.assertRaises(TypeError):
            q *= p
        with self.assertRaises(ValueError):
            q *= o
        with self.assertRaises(ValueError):
            q *= n
        with self.assertRaises(ValueError):
            q *= (np.random.rand(5) - 0.5)

        # q = Quat(a)
        # q *= p
        # self.assertIsInstance(q, Quat)
        # self.assertTrue(np.isclose(q.numpy(), Quat.hamilton_product(a, p)).all())
        # q = Quat(a)
        # q *= o
        # self.assertIsInstance(q, Quat)
        # self.assertTrue(np.isclose(q.numpy(), Quat.hamilton_product(a, o)).all())
        # q = Quat(a)
        # q *= n
        # self.assertIsInstance(q, Quat)
        # self.assertTrue(np.isclose(q.numpy(), Quat.hamilton_product(a, n)).all())


    def test_member_scalar_multiply(self):
        a = np.random.rand(4) - 0.5
        q = Quat(a)
        s_int = np.random.randint(-10, 10)
        s_float = np.random.rand(1)[0] - 0.5
        s_arr_onedim_single = np.random.rand(1) - 0.5
        s_arr_twodim_single = np.random.rand(1, 1) - 0.5
        p = Quat(np.random.rand(4) - 0.5)
        o = np.random.rand(4) - 0.5
        n = np.random.rand(1, 4) - 0.5

        # Quat *= scalars
        res = q.scalar_multiply(s_int)
        self.assertIsInstance(res, Quat)
        self.assertTrue(np.isclose(res.numpy(), a * s_int).all())
        res = q.scalar_multiply(s_float)
        self.assertIsInstance(res, Quat)
        self.assertTrue(np.isclose(res.numpy(), a * s_float).all())
        res = q.scalar_multiply(s_arr_onedim_single)
        self.assertIsInstance(res, Quat)
        self.assertTrue(np.isclose(res.numpy(), a * s_arr_onedim_single).all())
        res = q.scalar_multiply(s_arr_twodim_single)
        self.assertIsInstance(res, np.ndarray)
        for i, x in enumerate(res):
            self.assertIsInstance(x, Quat)
            self.assertTrue(np.isclose(x.numpy(), a * s_arr_twodim_single[i]).all())

        # Quat *= Quat should not be supported by scalar multiply *
        with self.assertRaises(TypeError):
            q.scalar_multiply(p)
        with self.assertRaises(ValueError):
            q.scalar_multiply(o)
        with self.assertRaises(ValueError):
            q.scalar_multiply(n)
        with self.assertRaises(ValueError):
            q.scalar_multiply((np.random.rand(5) - 0.5))


    def test_member_scalar_multiply_inplace(self):
        a = np.random.rand(4) - 0.5
        s_int = np.random.randint(-10, 10)
        s_float = np.random.rand(1)[0] - 0.5
        s_arr_onedim_single = np.random.rand(1) - 0.5
        s_arr_twodim_single = np.random.rand(1, 1) - 0.5
        p = Quat(np.random.rand(4) - 0.5)
        o = np.random.rand(4) - 0.5
        n = np.random.rand(1, 4) - 0.5
        m = np.random.rand(6) - 0.5
        l = np.random.rand(1, 6) - 0.5

        # Quat * Scalar
        q = Quat(a)
        q.scalar_multiply_(s_int)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), a * s_int).all())
        q = Quat(a)
        q.scalar_multiply_(s_float)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), a * s_float).all())
        q = Quat(a)
        q.scalar_multiply_(s_arr_onedim_single)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), a * s_arr_onedim_single).all())
        q = Quat(a)
        q.scalar_multiply_(s_arr_twodim_single)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), a * s_arr_twodim_single).all())

        # Quat * Quat not supported
        with self.assertRaises(TypeError):
            q.scalar_multiply_(p)
        with self.assertRaises(ValueError):
            q.scalar_multiply_(o)
        with self.assertRaises(ValueError):
            q.scalar_multiply_(n)
        with self.assertRaises(ValueError):
            q.scalar_multiply_(m)
        with self.assertRaises(ValueError):
            q.scalar_multiply_(l)

    ######################################
    #   End Testing of Scalar Multiply   #
    ######################################


    #########################################
    #   Start Testing of Hamilton Product   #
    #########################################

    def test_static_hamilton_product(self):
        q = Quat(np.random.rand(4) - 0.5)
        p = Quat(np.random.rand(4) - 0.5)
        a1, b1, c1, d1 = q.numpy()
        a2, b2, c2, d2 = p.numpy()
        # @see https://en.wikipedia.org/wiki/Quaternion
        res_gt = Quat(a=(a1*a2 - b1*b2 - c1*c2 - d1*d2),
                      b=(a1*b2 + b1*a2 + c1*d2 - d1*c2),
                      c=(a1*c2 - b1*d2 + c1*a2 + d1*b2),
                      d=(a1*d2 + b1*c2 - c1*b2 + d1*a2)
                      )
        res = Quat.hamilton_product(q, p)
        self.assertTrue(np.isclose(res, res_gt.numpy()).all())
        p = np.random.rand(10, 4) - 0.5
        res = Quat.hamilton_product(q, p)
        for i, x in enumerate(p):
            a2, b2, c2, d2 = x
            res_gt = Quat(a=(a1*a2 - b1*b2 - c1*c2 - d1*d2),
                      b=(a1*b2 + b1*a2 + c1*d2 - d1*c2),
                      c=(a1*c2 - b1*d2 + c1*a2 + d1*b2),
                      d=(a1*d2 + b1*c2 - c1*b2 + d1*a2)
                      )
            self.assertTrue(np.isclose(res[i], res_gt.numpy()).all())
        q = np.random.rand(10, 4) - 0.5
        p = np.random.rand(4) - 0.5
        a2, b2, c2, d2 = p
        res = Quat.hamilton_product(q, p)
        for i, x in enumerate(q):
            a1, b1, c1, d1 = x
            res_gt = Quat(a=(a1*a2 - b1*b2 - c1*c2 - d1*d2),
                      b=(a1*b2 + b1*a2 + c1*d2 - d1*c2),
                      c=(a1*c2 - b1*d2 + c1*a2 + d1*b2),
                      d=(a1*d2 + b1*c2 - c1*b2 + d1*a2)
                      )
            self.assertTrue(np.isclose(res[i], res_gt.numpy()).all())
        q = np.random.rand(10, 4) - 0.5
        p = np.random.rand(10, 4) - 0.5
        res = Quat.hamilton_product(q, p)
        for i, _ in enumerate(q):
            a1, b1, c1, d1 = q[i]
            a2, b2, c2, d2 = p[i]
            res_gt = Quat(a=(a1*a2 - b1*b2 - c1*c2 - d1*d2),
                      b=(a1*b2 + b1*a2 + c1*d2 - d1*c2),
                      c=(a1*c2 - b1*d2 + c1*a2 + d1*b2),
                      d=(a1*d2 + b1*c2 - c1*b2 + d1*a2)
                      )
            self.assertTrue(np.isclose(res[i], res_gt.numpy()).all())

        # Numpy arrays of Quat
        a = np.random.rand(10, 4) - 0.5
        q = np.asarray([Quat(elem) for elem in a])
        p = np.random.rand(4) - 0.5
        a2, b2, c2, d2 = p
        res = Quat.hamilton_product(q, p)
        for i, x in enumerate(a):
            a1, b1, c1, d1 = x
            res_gt = Quat(a=(a1*a2 - b1*b2 - c1*c2 - d1*d2),
                      b=(a1*b2 + b1*a2 + c1*d2 - d1*c2),
                      c=(a1*c2 - b1*d2 + c1*a2 + d1*b2),
                      d=(a1*d2 + b1*c2 - c1*b2 + d1*a2)
                      )
            self.assertTrue(np.isclose(res[i], res_gt.numpy()).all())
        a = np.random.rand(10, 4) - 0.5
        q = np.asarray([Quat(elem) for elem in a])
        b = np.random.rand(10, 4) - 0.5
        p = np.asarray([Quat(elem) for elem in b])
        res = Quat.hamilton_product(q, p)
        for i, _ in enumerate(q):
            a1, b1, c1, d1 = a[i]
            a2, b2, c2, d2 = b[i]
            res_gt = Quat(a=(a1*a2 - b1*b2 - c1*c2 - d1*d2),
                      b=(a1*b2 + b1*a2 + c1*d2 - d1*c2),
                      c=(a1*c2 - b1*d2 + c1*a2 + d1*b2),
                      d=(a1*d2 + b1*c2 - c1*b2 + d1*a2)
                      )
            self.assertTrue(np.isclose(res[i], res_gt.numpy()).all())

        # Test invalid values
        s_int = np.random.randint(-10, 10)
        s_float = np.random.rand(1)[0] - 0.5
        q = np.random.rand(5, 4)
        p = np.random.rand(2, 4)
        o = np.random.rand(3, 5)
        n = np.random.rand(3)
        with self.assertRaises(TypeError):
            Quat.hamilton_product(q, s_int)
        with self.assertRaises(TypeError):
            Quat.hamilton_product(q, s_float)
        with self.assertRaises(ValueError):
            Quat.hamilton_product(q, o)
        with self.assertRaises(ValueError):
            Quat.hamilton_product(n, n)


    def test_operator_matmul(self):
        q = Quat(np.random.rand(4) - 0.5)
        p = Quat(np.random.rand(4) - 0.5)
        o = np.random.rand(4) - 0.5
        n = np.random.rand(10, 4) - 0.5

        res = q @ p
        self.assertIsInstance(res, Quat)
        self.assertTrue(np.isclose(res.numpy(), Quat.hamilton_product(q, p)).all())
        res = q @ o
        self.assertIsInstance(res, Quat)
        self.assertTrue(np.isclose(res.numpy(), Quat.hamilton_product(q, o)).all())
        res = q @ n
        self.assertIsInstance(res, np.ndarray)
        for i, x in enumerate(res):
            self.assertIsInstance(x, Quat)
            self.assertTrue(np.isclose(x.numpy(), Quat.hamilton_product(q, n[i])).all())

        # Test invalid values
        s_int = np.random.randint(-10, 10)
        s_float = np.random.rand(1)[0] - 0.5
        arr_single_invalid = np.random.rand(1) - 0.5
        arr_multi_invalid = np.random.rand(10, 1) - 0.5
        with self.assertRaises(TypeError):
            q @ s_int
        with self.assertRaises(TypeError):
            q @ s_float
        with self.assertRaises(ValueError):
            q @ arr_single_invalid
        with self.assertRaises(ValueError):
            q @ arr_multi_invalid


    def test_operator_reverse_matmul(self):
        q = Quat(np.random.rand(4) - 0.5)
        p = Quat(np.random.rand(4) - 0.5)
        o = np.random.rand(4) - 0.5
        n = np.random.rand(10, 4) - 0.5

        res = p @ q
        self.assertIsInstance(res, Quat)
        self.assertTrue(np.isclose(res.numpy(), Quat.hamilton_product(p, q)).all())
        res = o @ q
        self.assertIsInstance(res, Quat)
        self.assertTrue(np.isclose(res.numpy(), Quat.hamilton_product(o, q)).all())
        res = n @ q
        self.assertIsInstance(res, np.ndarray)
        for i, x in enumerate(res):
            self.assertIsInstance(x, Quat)
            self.assertTrue(np.isclose(x.numpy(), Quat.hamilton_product(n[i], q)).all())

        # Test invalid values
        s_int = np.random.randint(-10, 10)
        s_float = np.random.rand(1)[0] - 0.5
        arr_single_invalid = np.random.rand(1) - 0.5
        arr_multi_invalid = np.random.rand(10, 1) - 0.5
        with self.assertRaises(TypeError):
            s_int @ q
        with self.assertRaises(TypeError):
            s_float @ q
        with self.assertRaises(ValueError):
            arr_single_invalid @ q
        with self.assertRaises(ValueError):
            arr_multi_invalid @ q


    def test_operator_inplace_matmul(self):
        a = np.random.rand(4) - 0.5
        p = Quat(np.random.rand(4) - 0.5)
        o = np.random.rand(4) - 0.5
        n = np.random.rand(1, 4) - 0.5

        # Quat *= Quat
        q = Quat(a)
        q @= p
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), Quat.hamilton_product(a, p)).all())
        q = Quat(a)
        q @= o
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), Quat.hamilton_product(a, o)).all())
        q = Quat(a)
        q @= n
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), Quat.hamilton_product(a, n)).all())

        # Test invalid values
        s_int = np.random.randint(-10, 10)
        s_float = np.random.rand(1)[0] - 0.5
        arr_single_invalid = np.random.rand(1) - 0.5
        arr_multi_invalid = np.random.rand(10, 1) - 0.5
        multi_quat = np.random.rand(10, 4) - 0.5
        q = Quat(a)
        with self.assertRaises(TypeError):
            q @= s_int
        with self.assertRaises(TypeError):
            q @= s_float
        with self.assertRaises(ValueError):
            q @= arr_single_invalid
        with self.assertRaises(ValueError):
            q @= arr_multi_invalid
        with self.assertRaises(ValueError):
            q @= multi_quat


    def test_member_hamilton(self):
        q = Quat(np.random.rand(4) - 0.5)
        p = Quat(np.random.rand(4) - 0.5)
        o = np.random.rand(4) - 0.5
        n = np.random.rand(10, 4) - 0.5

        res = q.hamilton_product(p)
        self.assertIsInstance(res, Quat)
        self.assertTrue(np.isclose(res.numpy(), Quat.hamilton_product(q, p)).all())
        res = q.hamilton_product(o)
        self.assertIsInstance(res, Quat)
        self.assertTrue(np.isclose(res.numpy(), Quat.hamilton_product(q, o)).all())
        res = q.hamilton_product(n)
        self.assertIsInstance(res, np.ndarray)
        for i, x in enumerate(res):
            self.assertIsInstance(x, Quat)
            self.assertTrue(np.isclose(x.numpy(), Quat.hamilton_product(q, n[i])).all())

        # Test invalid values
        s_int = np.random.randint(-10, 10)
        s_float = np.random.rand(1)[0] - 0.5
        arr_single_invalid = np.random.rand(1) - 0.5
        arr_multi_invalid = np.random.rand(10, 1) - 0.5
        with self.assertRaises(TypeError):
            q.hamilton_product(s_int)
        with self.assertRaises(TypeError):
            q.hamilton_product(s_float)
        with self.assertRaises(ValueError):
            q.hamilton_product(arr_single_invalid)
        with self.assertRaises(ValueError):
            q.hamilton_product(arr_multi_invalid)


    def test_member_inplace_hamilton(self):
        a = np.random.rand(4) - 0.5
        p = Quat(np.random.rand(4) - 0.5)
        o = np.random.rand(4) - 0.5
        n = np.random.rand(1, 4) - 0.5

        # Quat *= Quat
        q = Quat(a)
        q.hamilton_product_(p)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), Quat.hamilton_product(a, p)).all())
        q = Quat(a)
        q.hamilton_product_(o)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), Quat.hamilton_product(a, o)).all())
        q = Quat(a)
        q.hamilton_product_(n)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), Quat.hamilton_product(a, n)).all())

        # Test invalid values
        s_int = np.random.randint(-10, 10)
        s_float = np.random.rand(1)[0] - 0.5
        arr_single_invalid = np.random.rand(1) - 0.5
        arr_multi_invalid = np.random.rand(10, 1) - 0.5
        multi_quat = np.random.rand(10, 4) - 0.5
        q = Quat(a)
        with self.assertRaises(TypeError):
            q.hamilton_product_(s_int)
        with self.assertRaises(TypeError):
            q.hamilton_product_(s_float)
        with self.assertRaises(ValueError):
            q.hamilton_product_(arr_single_invalid)
        with self.assertRaises(ValueError):
            q.hamilton_product_(arr_multi_invalid)
        with self.assertRaises(ValueError):
            q.hamilton_product_(multi_quat)

    #######################################
    #   End Testing of Hamilton Product   #
    #######################################


    #################################
    #   Start Testing of Division   #
    #################################

    def test_operator_true_division(self):
        a = np.random.rand(4)
        q = Quat(a)
        s_int = np.random.randint(-10, 10)
        while s_int == 0:
            s_int = np.random.randint(-10, 10)
        s_int_zero = 0
        s_float = np.random.rand(1)[0] - 0.5
        while s_float == 0:
            s_float = np.random.rand(1)[0] - 0.5
        s_float_zero = 0.0
        s_arr_onedim_single = np.random.rand(1) - 0.5
        while np.isclose(s_arr_onedim_single, 0).any():
            s_arr_onedim_single = np.random.rand(1) - 0.5
        s_arr_onedim_single_zero = np.asarray([0])
        s_arr_twodim_single = np.random.rand(1, 1) - 0.5
        while np.isclose(s_arr_twodim_single, 0).any():
            s_arr_twodim_single = np.random.rand(1, 1) - 0.5
        s_arr_twodim_single_zero = np.asarray([[0]])
        s_arr_twodim_multi = np.random.rand(10, 1) - 0.5
        while np.isclose(s_arr_twodim_multi, 0).any():
            s_arr_twodim_multi = np.random.rand(10, 1) - 0.5
        s_arr_twodim_multi_zero = np.zeros((10, 1))

        # Division by scalars
        mul_res = q * s_int
        res = mul_res / s_int
        self.assertIsInstance(res, Quat)
        self.assertTrue(np.isclose(q.numpy(), res.numpy()).all())
        mul_res = q * s_float
        res = mul_res / s_float
        self.assertIsInstance(res, Quat)
        self.assertTrue(np.isclose(q.numpy(), res.numpy()).all())
        mul_res = q * s_arr_onedim_single
        res = mul_res / s_arr_onedim_single
        self.assertIsInstance(res, Quat)
        self.assertTrue(np.isclose(q.numpy(), res.numpy()).all())
        mul_res = q * s_arr_twodim_single
        self.assertIsInstance(mul_res, np.ndarray)
        res = mul_res[0] / s_arr_twodim_single
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(res.ndim == 1 and len(res) == 1)
        self.assertIsInstance(res[0], Quat)
        self.assertTrue(np.isclose(res[0].numpy(), q.numpy()).all())
        res = q / s_arr_twodim_multi
        self.assertIsInstance(res, np.ndarray)
        for i, x in enumerate(res):
            self.assertIsInstance(x, Quat)
            self.assertTrue(np.isclose(x.numpy(), q.numpy() / s_arr_twodim_multi[i]).all())

        # Division by zero scalar
        with self.assertRaises(ZeroDivisionError):
            q / s_int_zero
        with self.assertRaises(ZeroDivisionError):
            q / s_float_zero
        with self.assertRaises(ZeroDivisionError):
            q / s_arr_onedim_single_zero
        with self.assertRaises(ZeroDivisionError):
            q / s_arr_twodim_single_zero
        with self.assertRaises(ZeroDivisionError):
            q / s_arr_twodim_multi_zero

        # Division by other quaternion is not supported
        p = Quat(np.random.rand(4) - 0.5)
        with self.assertRaises(TypeError):
            q / p
        o = np.random.rand(4) - 0.5
        with self.assertRaises(TypeError):
            q / o
        n = np.random.rand(10, 4) - 0.5
        with self.assertRaises(TypeError):
            q / n

        # Invalid values
        b = np.random.rand(3)
        with self.assertRaises(ValueError):
            q / b
        b = np.random.rand(10, 5)
        with self.assertRaises(ValueError):
            q / b


    def test_operator_reverse_true_division(self):
        a = np.random.rand(4) - 0.5
        q = Quat(a)
        s_int = np.random.randint(-10, 10)
        while s_int == 0:
            s_int = np.random.randint(-10, 10)
        s_int_zero = 0
        s_float = np.random.rand(1)[0] - 0.5
        while s_float == 0:
            s_float = np.random.rand(1)[0] - 0.5
        s_float_zero = 0.0
        s_arr_onedim_single = np.random.rand(1) - 0.5
        while np.isclose(s_arr_onedim_single, 0).any():
            s_arr_onedim_single = np.random.rand(1) - 0.5
        s_arr_onedim_single_zero = np.asarray([0])
        s_arr_twodim_single = np.random.rand(1, 1) - 0.5
        while np.isclose(s_arr_twodim_single, 0).any():
            s_arr_twodim_single = np.random.rand(1, 1) - 0.5
        s_arr_twodim_single_zero = np.asarray([[0]])
        s_arr_twodim_multi = np.random.rand(10, 1) - 0.5
        while np.isclose(s_arr_twodim_multi, 0).any():
            s_arr_twodim_multi = np.random.rand(10, 1) - 0.5
        s_arr_twodim_multi_zero = np.zeros((10, 1))

        # Scalar divided by quaternion
        res = s_int / q
        self.assertIsInstance(res, Quat)
        self.assertTrue(np.isclose(res.numpy(), s_int * q.reciprocal().numpy()).all())
        res = s_float / q
        self.assertIsInstance(res, Quat)
        self.assertTrue(np.isclose(res.numpy(), s_float * q.reciprocal().numpy()).all())
        res = s_int_zero / q
        self.assertIsInstance(res, Quat)
        self.assertTrue(res.is_zero())
        res = s_float_zero / q
        self.assertIsInstance(res, Quat)
        self.assertTrue(res.is_zero())
        res = s_arr_onedim_single / q
        self.assertIsInstance(res, Quat)
        self.assertTrue(np.isclose(res.numpy(), s_arr_onedim_single * q.reciprocal().numpy()).all())
        res = s_arr_onedim_single_zero / q
        self.assertIsInstance(res, Quat)
        self.assertTrue(res.is_zero())
        res = s_arr_twodim_single / q
        self.assertIsInstance(res, np.ndarray)
        for i, x in enumerate(res):
            self.assertIsInstance(x, Quat)
            self.assertTrue(np.isclose(x.numpy(), (s_arr_twodim_single[i] * q.reciprocal()).numpy()).all())
        res = s_arr_twodim_single_zero / q
        self.assertIsInstance(res, np.ndarray)
        for i, x in enumerate(res):
            self.assertIsInstance(x, Quat)
            self.assertTrue(x.is_zero())
        res = s_arr_twodim_multi / q
        self.assertIsInstance(res, np.ndarray)
        for i, x in enumerate(res):
            self.assertIsInstance(x, Quat)
            self.assertTrue(np.isclose(x.numpy(), (s_arr_twodim_multi[i] * q.reciprocal()).numpy()).all())
        res = s_arr_twodim_multi_zero / q
        self.assertIsInstance(res, np.ndarray)
        for i, x in enumerate(res):
            self.assertIsInstance(x, Quat)
            self.assertTrue(x.is_zero())

        # Invalid arguments
        with self.assertRaises(ValueError):
            np.random.rand(3) / q
        with self.assertRaises(ValueError):
            np.random.rand(10, 5) / q


    def test_operator_inplace_true_division(self):
        a = np.random.rand(4) - 0.5
        q = Quat(a)
        s_int = np.random.randint(-10, 10)
        while s_int == 0:
            s_int = np.random.randint(-10, 10)
        s_int_zero = int(0)
        s_float = np.random.rand(1)[0] - 0.5
        while math.isclose(s_float,  0):
            s_float = np.random.rand(1)[0] - 0.5
        s_float_zero = 0
        s_arr_onedim = np.random.rand(1) - 0.5
        s_arr_twodim = np.random.rand(1, 1) - 0.5
        q /= s_int
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), (Quat(a) / s_int).numpy()).all())
        q = Quat(a)
        q /= s_float
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), (Quat(a) / s_float).numpy()).all())
        q = Quat(a)
        q /= s_arr_onedim
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), (Quat(a) / s_arr_onedim).numpy()).all())
        q = Quat(a)
        q /= s_arr_twodim
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), (Quat(a) / s_arr_twodim)[0].numpy()).all())
        q = Quat(a)
        with self.assertRaises(ZeroDivisionError):
            q /= s_int_zero
        with self.assertRaises(ZeroDivisionError):
            q /= s_float_zero

        # Invalid entries
        p = Quat(1,2,3,4)
        with self.assertRaises(TypeError):
            q /= p
        p = np.random.rand(2)
        with self.assertRaises(ValueError):
            q /= p
        p = np.random.rand(3, 1)
        with self.assertRaises(ValueError):
            q /= p
        p = np.random.rand(5, 3)
        with self.assertRaises(ValueError):
            q /= p


    def test_member_div(self):
        a = np.random.rand(4) - 0.5
        q = Quat(a)
        b = np.random.rand(4) - 0.5
        p = Quat(b)
        q_multi = np.random.rand(10, 4) - 0.5
        p_multi = np.random.rand(10, 4) - 0.5
        s_int = np.random.randint(-10, 10)
        while s_int == 0:
            s_int = np.random.randint(-10, 10)
        s_int_zero = int(0)
        s_float = np.random.rand(1)[0] - 0.5
        while s_float == 0.0:
            s_float = np.random.rand(1)[0] - 0.5
        s_float_zero = float(0)
        s_arr_onedim = np.random.rand(1) - 0.5
        s_arr_twodim_single = np.random.rand(1, 1) - 0.5
        s_arr_twodim_multi = np.random.rand(10, 1) - 0.5

        # Quaternion div scalar
        lh = q.div_lhand(s_int)
        rh = q.div_rhand(s_int)
        self.assertIsInstance(lh, Quat)
        self.assertIsInstance(rh, Quat)
        self.assertTrue(np.isclose(lh.numpy(), (q / s_int).numpy()).all())
        self.assertTrue(np.isclose(lh.numpy(), rh.numpy()).all())
        lh = q.div_lhand(s_float)
        rh = q.div_rhand(s_float)
        self.assertIsInstance(lh, Quat)
        self.assertIsInstance(rh, Quat)
        self.assertTrue(np.isclose(lh.numpy(), (q / s_float).numpy()).all())
        self.assertTrue(np.isclose(lh.numpy(), rh.numpy()).all())
        lh = q.div_lhand(s_arr_onedim)
        rh = q.div_rhand(s_arr_onedim)
        self.assertIsInstance(lh, Quat)
        self.assertIsInstance(rh, Quat)
        self.assertTrue(np.isclose(lh.numpy(), (q / s_arr_onedim).numpy()).all())
        self.assertTrue(np.isclose(lh.numpy(), rh.numpy()).all())
        lh = q.div_lhand(s_arr_twodim_single)
        rh = q.div_rhand(s_arr_twodim_single)
        self.assertIsInstance(lh, np.ndarray)
        self.assertIsInstance(rh, np.ndarray)
        for i, x in enumerate(lh):
            self.assertIsInstance(x, Quat)
            self.assertIsInstance(rh[i], Quat)
            self.assertTrue(np.isclose(x.numpy(), (q / s_arr_twodim_single[i]).numpy()).all())
            self.assertTrue(np.isclose(x.numpy(), rh[i].numpy()).all())
        lh = q.div_lhand(s_arr_twodim_multi)
        rh = q.div_rhand(s_arr_twodim_multi)
        self.assertIsInstance(lh, np.ndarray)
        self.assertIsInstance(rh, np.ndarray)
        for i, x in enumerate(lh):
            self.assertIsInstance(x, Quat)
            self.assertIsInstance(rh[i], Quat)
            self.assertTrue(np.isclose(x.numpy(), (q / s_arr_twodim_multi[i]).numpy()).all())
            self.assertTrue(np.isclose(x.numpy(), rh[i].numpy()).all())

        # Check zero division errors
        with self.assertRaises(ZeroDivisionError):
            q.div_lhand(s_int_zero)
        with self.assertRaises(ZeroDivisionError):
            q.div_rhand(s_int_zero)
        with self.assertRaises(ZeroDivisionError):
            q.div_lhand(s_float_zero)
        with self.assertRaises(ZeroDivisionError):
            q.div_rhand(s_float_zero)
        with self.assertRaises(ZeroDivisionError):
            q.div_lhand(np.asarray([0]))
        with self.assertRaises(ZeroDivisionError):
            q.div_rhand(np.asarray([0]))
        with self.assertRaises(ZeroDivisionError):
            q.div_lhand(np.asarray([[0], [1]]))
        with self.assertRaises(ZeroDivisionError):
            q.div_rhand(np.asarray([[1], [0]]))

        # Quaternion div single quaternion
        r = q @ p
        q_div = r.div_rhand(p)
        p_div = r.div_lhand(q)
        self.assertIsInstance(q_div, Quat)
        self.assertIsInstance(p_div, Quat)
        self.assertTrue(np.isclose(q_div.numpy(), q.numpy()).all())
        self.assertTrue(np.isclose(p_div.numpy(), p.numpy()).all())

        # Quaternion div multiple quaternions
        q = np.random.rand(10, 4) - 0.5
        for x in q:
            # assume r = n @ p where p is unknown and n is known
            r = Quat(x)
            n = np.random.rand(10, 4) - 0.5
            # calculate p by left hand division: p = n^-1 @ r
            p = r.div_lhand(n)
            self.assertIsInstance(p, np.ndarray)
            # make sure that n @ p again equals r
            r_hat = Quat.hamilton_product(n, p)
            for y in r_hat:
                self.assertIsInstance(y, np.ndarray)
                self.assertTrue(np.isclose(y, r.numpy()).all())
        q = np.random.rand(10, 4) - 0.5
        for x in q:
            # assume r = n @ p where p is known and n is unknown
            r = Quat(x)
            p = np.random.rand(10, 4) - 0.5
            # calculate n by right hand division n = r @ p^-1
            n = r.div_rhand(p)
            self.assertIsInstance(n, np.ndarray)
            # make sure that n @ p again equals r
            r_hat = Quat.hamilton_product(n, p)
            for y in r_hat:
                self.assertIsInstance(y, np.ndarray)
                self.assertTrue(np.isclose(y, r.numpy()).all())


    def test_member_inplace_div(self):
        a = np.random.rand(4) - 0.5
        b = np.random.rand(4) - 0.5
        s_arr_onedim_single = np.random.rand(1) - 0.5
        s_arr_twodim_single = np.random.rand(1,1) - 0.5

        s_int = np.random.randint(-20, 20)
        while s_int == 0:
            s_int = np.random.randint(-20, 20)
        s_int_zero = int(0)
        s_float = np.random.rand(1)[0] - 0.5
        while np.isclose(s_float, 0):
            s_float = np.random.rand(1)[0] - 0.5
        s_float_zero = float(0)

        # Quaternion div scalar
        ### lhand
        q = Quat(a)
        q.div_lhand_(s_int)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), Quat(a).div_lhand(s_int).numpy()).all())
        q = Quat(a)
        q.div_lhand_(s_float)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), Quat(a).div_lhand(s_float).numpy()).all())
        q = Quat(a)
        with self.assertRaises(ZeroDivisionError):
            q.div_lhand_(s_int_zero)
        with self.assertRaises(ZeroDivisionError):
            q.div_lhand_(s_float_zero)
        q = Quat(a)
        q.div_lhand_(s_arr_onedim_single)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), Quat(a).div_lhand(s_arr_onedim_single).numpy()).all())
        q = Quat(a)
        q.div_lhand_(s_arr_twodim_single)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), Quat(a).div_lhand(s_arr_twodim_single)[0].numpy()).all())
        q = Quat(a)
        with self.assertRaises(ZeroDivisionError):
            q.div_lhand_(np.asarray([0]))
        with self.assertRaises(ZeroDivisionError):
            q.div_lhand_(np.asarray([[0]]))
        with self.assertRaises(ValueError):
            q.div_lhand_(np.asarray([1,2]))
        with self.assertRaises(ValueError):
            q.div_lhand_(np.asarray([[1],[2]]))
        with self.assertRaises(ValueError):
            q.div_lhand_(np.asarray([[1,2],[3,4]]))
        ### rhand
        q = Quat(a)
        q.div_rhand_(s_int)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), Quat(a).div_rhand(s_int).numpy()).all())
        q = Quat(a)
        q.div_rhand_(s_float)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), Quat(a).div_rhand(s_float).numpy()).all())
        q = Quat(a)
        with self.assertRaises(ZeroDivisionError):
            q.div_rhand_(s_int_zero)
        with self.assertRaises(ZeroDivisionError):
            q.div_rhand_(s_float_zero)
        q = Quat(a)
        q.div_rhand_(s_arr_onedim_single)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), Quat(a).div_rhand(s_arr_onedim_single).numpy()).all())
        q = Quat(a)
        q.div_rhand_(s_arr_twodim_single)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), Quat(a).div_rhand(s_arr_twodim_single)[0].numpy()).all())
        q = Quat(a)
        with self.assertRaises(ZeroDivisionError):
            q.div_rhand_(np.asarray([0]))
        with self.assertRaises(ZeroDivisionError):
            q.div_rhand_(np.asarray([[0]]))
        with self.assertRaises(ValueError):
            q.div_rhand_(np.asarray([1,2]))
        with self.assertRaises(ValueError):
            q.div_rhand_(np.asarray([[1],[2]]))
        with self.assertRaises(ValueError):
            q.div_rhand_(np.asarray([[1,2],[3,4]]))

        # Quaternion div quaternion
        q = Quat(a)
        p = Quat(b)
        q.div_lhand_(p)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), Quat(a).div_lhand(p).numpy()).all())
        q = Quat(a)
        q.div_rhand_(p)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), Quat(a).div_rhand(p).numpy()).all())
        q = Quat(a)
        with self.assertRaises(ZeroDivisionError):
            q.div_lhand_(Quat())
        with self.assertRaises(ZeroDivisionError):
            q.div_rhand_(Quat())

        # Quaternion div numpy quaternion
        p = np.random.rand(4) - 0.5
        q = Quat(a)
        q.div_lhand_(p)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), Quat(a).div_lhand(p).numpy()).all())
        q = Quat(a)
        q.div_rhand_(p)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), Quat(a).div_rhand(p).numpy()).all())
        p = np.random.rand(1, 4) - 0.5
        q = Quat(a)
        q.div_lhand_(p)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), Quat(a).div_lhand(p)[0].numpy()).all())
        q = Quat(a)
        q.div_rhand_(p)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), Quat(a).div_rhand(p)[0].numpy()).all())
        with self.assertRaises(ZeroDivisionError):
            q.div_lhand_(np.asarray([0,0,0,0]))
        with self.assertRaises(ZeroDivisionError):
            q.div_rhand_(np.asarray([[0,0,0,0]]))
        with self.assertRaises(ValueError):
            q.div_lhand_(np.asarray([0,1]))
        with self.assertRaises(ValueError):
            q.div_rhand_(np.asarray([0,1]))
        with self.assertRaises(ValueError):
            q.div_lhand_(np.asarray([[0,1,0,0], [0,1,1,1]]))
        with self.assertRaises(ValueError):
            q.div_rhand_(np.asarray([[0,1,0,0], [0,1,1,1]]))
        with self.assertRaises(ValueError):
            q.div_lhand_(np.asarray([[0,1,3]]))
        with self.assertRaises(ValueError):
            q.div_rhand_(np.asarray([[0,1,3]]))


    def test_static_div(self):
        # Quaternion(s) / scalar
        s_int = np.random.randint(-10, 10)
        while s_int == 0:
            s_int = np.random.randint(-10, 10)
        s_float = np.random.rand(1)[0] - 0.5
        while math.isclose(s_float, 0):
            s_float = np.random.rand(1)[0] - 0.5
        q = Quat(np.random.rand(4) - 0.5)
        res_lh = Quat.div_lhand(q, s_int)
        res_rh = Quat.div_rhand(q, s_int)
        self.assertTrue(np.isclose(res_lh, res_rh).all())
        self.assertTrue(np.isclose(res_lh, Quat.scalar_multiply(q, (1/s_int))).all())
        q = Quat(np.random.rand(4) - 0.5)
        res_lh = Quat.div_lhand(q, s_float)
        res_rh = Quat.div_rhand(q, s_float)
        self.assertTrue(np.isclose(res_lh, res_rh).all())
        self.assertTrue(np.isclose(res_lh, Quat.scalar_multiply(q, (1/s_float))).all())

        s_int_zero = int(0)
        s_float_zero = float(0)
        with self.assertRaises(ZeroDivisionError):
            Quat.div_lhand(Quat(1), s_int_zero)
        with self.assertRaises(ZeroDivisionError):
            Quat.div_lhand(Quat(1), s_float_zero)
        with self.assertRaises(ZeroDivisionError):
            Quat.div_rhand(Quat(1), s_int_zero)
        with self.assertRaises(ZeroDivisionError):
            Quat.div_rhand(Quat(1), s_float_zero)

        a = np.random.rand(10, 4) - 0.5
        res = Quat.div_lhand(a, s_int)
        self.assertTrue(np.isclose(res, Quat.scalar_multiply(a, (1/s_int))).all())
        self.assertTrue(np.isclose(Quat.div_rhand(a, s_int), res).all())
        res = Quat.div_lhand(a, s_float)
        self.assertTrue(np.isclose(res, Quat.scalar_multiply(a, (1/s_float))).all())
        self.assertTrue(np.isclose(Quat.div_rhand(a, s_float), res).all())

        # Scalar / quaternion(s)
        q = Quat(np.random.rand(4) - 0.5)
        while q.is_zero():
            q = Quat(np.random.rand(4) - 0.5)
        p = np.asarray([[1,2,3,-4],
                        [-0.5, 2, -1, -3],
                        [3,4,5,6]])
        q_zero = Quat()
        p_zero = np.asarray([[1,2,3,-4],
                             [0,0,0,0],
                             [3,4,5,6]])
        res = Quat.div_lhand(s_int, q)
        self.assertTrue(np.isclose(res, Quat.scalar_multiply(q.reciprocal(), s_int)).all())
        self.assertTrue(np.isclose(res, Quat.div_rhand(s_int, q)).all())
        res = Quat.div_lhand(s_float, q)
        self.assertTrue(np.isclose(res, Quat.scalar_multiply(q.reciprocal(), s_float)).all())
        self.assertTrue(np.isclose(res, Quat.div_rhand(s_float, q)).all())
        res = Quat.div_lhand(s_int, p)
        self.assertTrue(np.isclose(res, Quat.scalar_multiply(Quat.reciprocal(p), s_int)).all())
        self.assertTrue(np.isclose(res, Quat.div_rhand(s_int, p)).all())
        res = Quat.div_lhand(s_float, p)
        self.assertTrue(np.isclose(res, Quat.scalar_multiply(Quat.reciprocal(p), s_float)).all())
        self.assertTrue(np.isclose(res, Quat.div_rhand(s_float, p)).all())
        with self.assertRaises(ZeroDivisionError):
            Quat.div_lhand(s_int, q_zero)
        with self.assertRaises(ZeroDivisionError):
            Quat.div_lhand(s_float, q_zero)
        with self.assertRaises(ZeroDivisionError):
            Quat.div_rhand(s_int, q_zero)
        with self.assertRaises(ZeroDivisionError):
            Quat.div_rhand(s_float, q_zero)
        with self.assertRaises(ZeroDivisionError):
            Quat.div_lhand(s_int, p_zero)
        with self.assertRaises(ZeroDivisionError):
            Quat.div_lhand(s_float, p_zero)
        with self.assertRaises(ZeroDivisionError):
            Quat.div_rhand(s_int, p_zero)
        with self.assertRaises(ZeroDivisionError):
            Quat.div_rhand(s_float, p_zero)

        # Division with scalar represented as np array
        single_scalar = np.random.rand(1) - 0.5
        multiple_scalars = np.random.rand(10, 1) - 0.5
        # check providing both results in TypeError
        with self.assertRaises(TypeError):
            Quat.div_lhand(single_scalar, multiple_scalars)
        with self.assertRaises(TypeError):
            Quat.div_lhand(multiple_scalars, single_scalar)
        with self.assertRaises(TypeError):
            Quat.div_lhand(single_scalar, single_scalar)
        with self.assertRaises(TypeError):
            Quat.div_lhand(multiple_scalars, multiple_scalars)
        with self.assertRaises(TypeError):
            Quat.div_rhand(single_scalar, multiple_scalars)
        with self.assertRaises(TypeError):
            Quat.div_rhand(multiple_scalars, single_scalar)
        with self.assertRaises(TypeError):
            Quat.div_rhand(single_scalar, single_scalar)
        with self.assertRaises(TypeError):
            Quat.div_rhand(multiple_scalars, multiple_scalars)
        # Quat / np scalar
        a = np.random.rand(4) - 0.5
        res = Quat.div_lhand(a, single_scalar)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(res.ndim == 1 and len(res) == 4)
        self.assertTrue(np.isclose(res, Quat.scalar_multiply(a, (1/single_scalar))).all())
        res = Quat.div_rhand(a, single_scalar)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(res.ndim == 1 and len(res) == 4)
        self.assertTrue(np.isclose(res, Quat.scalar_multiply(a, (1/single_scalar))).all())
        res = Quat.div_lhand(a, multiple_scalars)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(res.ndim == 2 and res.shape[1] == 4)
        self.assertTrue(np.isclose(res, Quat.scalar_multiply(a, (1/multiple_scalars))).all())
        res = Quat.div_rhand(a, multiple_scalars)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(res.ndim == 2 and res.shape[1] == 4)
        self.assertTrue(np.isclose(res, Quat.scalar_multiply(a, (1/multiple_scalars))).all())
        # np scalar / Quat
        a = np.random.rand(4) - 0.5
        while Quat(a).is_zero():
            a = np.random.rand(4) - 0.5
        res = Quat.div_lhand(single_scalar, a)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(res.ndim == 1 and len(res) == 4)
        self.assertTrue(np.isclose(res, Quat.scalar_multiply(Quat.reciprocal(a), single_scalar)).all())
        res = Quat.div_rhand(single_scalar, a)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(res.ndim == 1 and len(res) == 4)
        self.assertTrue(np.isclose(res, Quat.scalar_multiply(Quat.reciprocal(a), single_scalar)).all())
        res = Quat.div_lhand(multiple_scalars, a)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(res.ndim == 2 and res.shape[1] == 4)
        self.assertTrue(np.isclose(res, Quat.scalar_multiply(Quat.reciprocal(a), multiple_scalars)).all())
        res = Quat.div_rhand(multiple_scalars, a)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(res.ndim == 2 and res.shape[1] == 4)
        self.assertTrue(np.isclose(res, Quat.scalar_multiply(Quat.reciprocal(a), multiple_scalars)).all())
        with self.assertRaises(ZeroDivisionError):
            Quat.div_lhand(a, np.asarray([0]))
        with self.assertRaises(ZeroDivisionError):
            Quat.div_lhand(a, np.asarray([[0], [0]]))
        with self.assertRaises(ZeroDivisionError):
            Quat.div_lhand(a, np.asarray([[1], [2], [0], [4]]))
        with self.assertRaises(ZeroDivisionError):
            Quat.div_lhand(np.asarray([1]), Quat())
        with self.assertRaises(ZeroDivisionError):
            Quat.div_rhand(a, np.asarray([0]))
        with self.assertRaises(ZeroDivisionError):
            Quat.div_rhand(a, np.asarray([[0], [0]]))
        with self.assertRaises(ZeroDivisionError):
            Quat.div_rhand(a, np.asarray([[1], [2], [0], [4]]))
        with self.assertRaises(ZeroDivisionError):
            Quat.div_rhand(np.asarray([1]), Quat())


        # Single quaternion / single quaternion
        a = np.random.rand(4) - 0.5
        b = np.random.rand(4) - 0.5
        q = Quat(a)
        p = Quat(b)
        res = Quat.div_lhand(q, p)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(res.ndim == 1)
        res_member = q.div_lhand(p)
        self.assertTrue(np.isclose(res, res_member.numpy()).all())
        res_np = Quat.div_lhand(a, b)
        self.assertTrue(np.isclose(res, res_np).all())
        res = Quat.div_rhand(q, p)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(res.ndim == 1)
        res_member = q.div_rhand(p)
        self.assertTrue(np.isclose(res, res_member.numpy()).all())
        res_np = Quat.div_rhand(a, b)
        self.assertTrue(np.isclose(res, res_np).all())

        # Single quaternion / multiple quaternions
        a = np.random.rand(4) - 0.5
        b = np.random.rand(10, 4) - 0.5
        res_lh = Quat.div_lhand(a, b)
        res_rh = Quat.div_rhand(a, b)
        for i, lh in enumerate(res_lh):
            self.assertTrue(np.isclose(lh, Quat(a).div_lhand(b[i]).numpy()).all())
        for i, rh in enumerate(res_rh):
            self.assertTrue(np.isclose(rh, Quat(a).div_rhand(b[i]).numpy()).all())
        b = np.asarray([[1,2,3,4], [0,0,0,0], [0,0,0,1]])
        with self.assertRaises(ZeroDivisionError):
            Quat.div_lhand(a, b)
        with self.assertRaises(ZeroDivisionError):
            Quat.div_rhand(a, b)

        # Multiple quaternions / single quaternion
        a = np.random.rand(10, 4) - 0.5
        b = np.random.rand(4) - 0.5
        while Quat(b).is_zero():
            b = np.random.rand(4) - 0.5
        res_lh = Quat.div_lhand(a, b)
        res_rh = Quat.div_rhand(a, b)
        for i, lh in enumerate(res_lh):
            self.assertTrue(np.isclose(lh, Quat(a[i]).div_lhand(b).numpy()).all())
        for i, rh in enumerate(res_rh):
            self.assertTrue(np.isclose(rh, Quat(a[i]).div_rhand(b).numpy()).all())
        with self.assertRaises(ZeroDivisionError):
            Quat.div_lhand(a, Quat())
        with self.assertRaises(ZeroDivisionError):
            Quat.div_lhand(a, Quat().numpy())
        with self.assertRaises(ZeroDivisionError):
            Quat.div_rhand(a, Quat())
        with self.assertRaises(ZeroDivisionError):
            Quat.div_rhand(a, Quat().numpy())

        # Multiple quaternions / multiple quaternions
        a = np.random.rand(10, 4) - 0.5
        b = np.random.rand(10, 4) - 0.5
        res_lh = Quat.div_lhand(a, b)
        res_rh = Quat.div_rhand(a, b)
        for i, lh in enumerate(res_lh):
            self.assertTrue(np.isclose(lh, Quat.div_lhand(a[i], b[i])).all())
        for i, rh in enumerate(res_rh):
            self.assertTrue(np.isclose(rh, Quat.div_rhand(a[i], b[i])).all())

    ###############################
    #   End Testing of Division   #
    ###############################


    ############################
    #   Start Testing of Pow   #
    ############################

    def test_static_pow(self):
        for _ in range(10):
            a = np.random.rand(4) - 0.5
            q = Quat(a)
            power = np.random.randint(1, 20)
            res = Quat.pow(q, power)
            res_gt = np.copy(a)
            while power > 1:
                res_gt = Quat.hamilton_product(res_gt, a)
                power -= 1
            self.assertTrue(np.isclose(res, res_gt).all())
        for _ in range(10):
            a = np.random.rand(10, 4) - 0.5
            power = np.random.randint(1, 20)
            res = Quat.pow(a, power)
            for i, x in enumerate(a):
                res_gt = np.copy(x)
                power_iter = power
                while power_iter > 1:
                    res_gt = Quat.hamilton_product(res_gt, x)
                    power_iter -= 1
                self.assertTrue(np.isclose(res[i], res_gt).all())
        q = Quat(np.random.rand(4) - 0.5)
        res = Quat.pow(q, 0)
        self.assertTrue(np.isclose(res, np.asarray([1,0,0,0])).all())

        # Test invalid parameters
        q = Quat()
        with self.assertRaises(TypeError):
            Quat.pow(q, 2.3)
        with self.assertRaises(TypeError):
            Quat.pow(q, 2.0)
        with self.assertRaises(ValueError):
            Quat.pow(q, -1)
        with self.assertRaises(TypeError):
            Quat.pow(3, q)


    def test_operator_pow(self):
        for _ in range(10):
            a = np.random.rand(4) - 0.5
            q = Quat(a)
            power = np.random.randint(1, 20)
            res = q ** power
            self.assertIsInstance(res, Quat)
            res_gt = np.copy(a)
            while power > 1:
                res_gt = Quat.hamilton_product(res_gt, a)
                power -= 1
            self.assertTrue(np.isclose(res.numpy(), res_gt).all())
        q = Quat(np.random.rand(4) - 0.5)
        res = q ** 0
        self.assertTrue(np.isclose(res.numpy(), np.asarray([1,0,0,0])).all())

        # Test invalid parameters
        q = Quat()
        with self.assertRaises(TypeError):
            q ** 2.3
        with self.assertRaises(TypeError):
            q ** 2.0
        with self.assertRaises(ValueError):
            q ** -1


    def test_member_pow(self):
        for _ in range(10):
            a = np.random.rand(4) - 0.5
            q = Quat(a)
            power = np.random.randint(1, 20)
            res = q.pow(power)
            self.assertIsInstance(res, Quat)
            res_gt = np.copy(a)
            while power > 1:
                res_gt = Quat.hamilton_product(res_gt, a)
                power -= 1
            self.assertTrue(np.isclose(res.numpy(), res_gt).all())
        q = Quat(np.random.rand(4) - 0.5)
        res = q.pow(0)
        self.assertTrue(np.isclose(res.numpy(), np.asarray([1,0,0,0])).all())

        # Test invalid parameters
        q = Quat()
        with self.assertRaises(TypeError):
            q.pow(2.3)
        with self.assertRaises(TypeError):
            q.pow(2.0)
        with self.assertRaises(ValueError):
            q.pow(-1)


    def test_member_inplace_pow(self):
        for _ in range(10):
            a = np.random.rand(4) - 0.5
            q = Quat(a)
            power = np.random.randint(1, 20)
            q.pow_(power)
            self.assertIsInstance(q, Quat)
            res_gt = np.copy(a)
            while power > 1:
                res_gt = Quat.hamilton_product(res_gt, a)
                power -= 1
            self.assertTrue(np.isclose(q.numpy(), res_gt).all())

        q = Quat(np.random.rand(4) - 0.5)
        q.pow_(0)
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.numpy(), np.asarray([1,0,0,0])).all())

        # Test invalid parameters
        q = Quat()
        with self.assertRaises(TypeError):
            q.pow_(2.3)
        with self.assertRaises(TypeError):
            q.pow_(2.0)
        with self.assertRaises(ValueError):
            q.pow_(-1)

    ##########################
    #   End Testing of Pow   #
    ##########################


    ###################################
    #   Start Testing of Reciprocal   #
    ###################################

    def test_static_reciprocal(self):
        a = np.random.rand(4) - 0.5
        q = Quat(a)
        q_rec = Quat.reciprocal(q)
        self.assertTrue(np.isclose(Quat.hamilton_product(q_rec, q), np.asarray([1,0,0,0])).all())
        self.assertTrue(np.isclose(Quat.hamilton_product(q, q_rec), np.asarray([1,0,0,0])).all())
        q = np.random.rand(10, 4) - 0.5
        q_rec = Quat.reciprocal(q)
        res = Quat.hamilton_product(q_rec, q)
        for x in res:
            self.assertTrue(np.isclose(x, np.asarray([1,0,0,0])).all())
        res = Quat.hamilton_product(q, q_rec)
        for x in res:
            self.assertTrue(np.isclose(x, np.asarray([1,0,0,0])).all())
        q = np.zeros(4)
        with self.assertRaises(ZeroDivisionError):
            Quat.reciprocal(q)
        q = np.zeros((10,4))
        with self.assertRaises(ZeroDivisionError):
            Quat.reciprocal(q)
        q = np.asarray([[1,2,3,4], [0,0,0,0], [4,3,2,1]])
        with self.assertRaises(ZeroDivisionError):
            Quat.reciprocal(q)

        # Invalid values
        s_int = np.random.randint(-10, 10)
        s_float = np.random.rand(1)[0] - 0.5
        with self.assertRaises(TypeError):
            Quat.reciprocal(s_int)
        with self.assertRaises(TypeError):
            Quat.reciprocal(s_float)
        with self.assertRaises(ValueError):
            Quat.reciprocal(np.random.rand(6) - 0.5)
        with self.assertRaises(ValueError):
            Quat.reciprocal(np.random.rand(3, 6) - 0.5)


    def test_member_reciprocal(self):
        a = np.random.rand(4) - 0.5
        q = Quat(a)
        rec = q.reciprocal()
        self.assertIsInstance(rec, Quat)
        self.assertTrue(np.isclose(Quat.hamilton_product(q, rec), np.asarray([1,0,0,0])).all())
        self.assertTrue(np.isclose(Quat.hamilton_product(rec, q), np.asarray([1,0,0,0])).all())
        q = Quat()
        with self.assertRaises(ZeroDivisionError):
            q.reciprocal()


    def test_member_inplace_reciprocal(self):
        a = np.random.rand(4) - 0.5
        q = Quat(a)
        q.reciprocal_()
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(Quat.hamilton_product(q, a), np.asarray([1,0,0,0])).all())
        self.assertTrue(np.isclose(Quat.hamilton_product(a, q), np.asarray([1,0,0,0])).all())
        q = Quat()
        with self.assertRaises(ZeroDivisionError):
            q.reciprocal_()

    #################################
    #   End Testing of Reciprocal   #
    #################################


    #############################
    #   Start Testing of Norm   #
    #############################

    def test_static_norm(self):
        a = np.random.rand(4) - 0.5
        b = Quat(a)
        self.assertTrue(np.isclose(np.linalg.norm(a), Quat.norm(b)), f"{np.linalg.norm(a)} vs {Quat.norm(b)}")
        self.assertTrue(np.isclose(np.linalg.norm(a), Quat.norm(a)))
        c = np.random.rand(10, 4) - 0.5
        self.assertTrue(np.allclose(np.linalg.norm(c, axis=1).reshape(-1, 1), Quat.norm(c)), f"{np.linalg.norm(c, axis=1).reshape(-1, 1)} vs {Quat.norm(c)}")


    def test_member_norm(self):
        a = np.random.rand(4) - 0.5
        q = Quat(a)
        self.assertTrue(np.allclose(np.linalg.norm(a), q.norm()))


    def test_static_normalize(self):
        a = np.random.rand(4) - 0.5
        q = Quat(a)
        b = np.random.rand(10,4) - 0.5
        normalized = Quat.normalize(q)
        self.assertIsInstance(normalized, np.ndarray)
        self.assertTrue(np.isclose(np.linalg.norm(normalized), 1))
        normalized = Quat.normalize(b)
        self.assertIsInstance(normalized, np.ndarray)
        for x in normalized:
            self.assertTrue(np.isclose(np.linalg.norm(x), 1))


    def test_member_normalize(self):
        a = np.random.rand(4) - 0.5
        q = Quat(a)
        p = q.normalize()
        self.assertIsInstance(p, Quat)
        self.assertTrue(np.isclose(p.norm(), 1))


    def test_member_inplace_normalize(self):
        a = np.random.rand(4) - 0.5
        q = Quat(a)
        q.normalize_()
        self.assertIsInstance(q, Quat)
        self.assertTrue(np.isclose(q.norm(), 1))


    def test_normalize_value_error_on_zero(self):
        q = Quat(0, 0, 0, 0)
        self.assertRaises(ValueError, Quat.normalize, q)
        self.assertRaises(ValueError, q.normalize)
        self.assertRaises(ValueError, q.normalize_)
        self.assertRaises(ValueError, Quat.normalize, np.zeros((10, 4)))
        a = np.asarray([[1,1,1,1], [0,0,0,0]])
        self.assertRaises(ValueError, Quat.normalize, a)

    ###########################
    #   End Testing of Norm   #
    ###########################


    ####################################
    #   Start Testing of Conjugation   #
    ####################################

    def test_static_conjugate(self):
        a = np.random.rand(4) - 0.5
        b = np.random.rand(10, 4) - 0.5
        conj_arr = np.asarray([1, -1, -1, -1])
        self.assertTrue((a*conj_arr == Quat.conjugate(a)).all())
        self.assertTrue((b*conj_arr == Quat.conjugate(b)).all())


    def test_member_conjugate(self):
        a = np.random.rand(4) - 0.5
        b = Quat(a)
        conj_arr = np.asarray([1, -1, -1, -1])
        c = b.conjugate()
        self.assertIsInstance(c, Quat)
        self.assertTrue((a*conj_arr == c.numpy()).all())
        b.conjugate_()
        self.assertIsInstance(b, Quat)
        self.assertTrue((a*conj_arr == b.numpy()).all())


    ##################################
    #   End Testing of Conjugation   #
    ##################################


if __name__ == '__main__':
    unittest.main()
