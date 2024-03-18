import unittest
import numpy as np
from quat import Quat

class TestQuat(unittest.TestCase):

    def test_init_from_np_arr(self):
        a = np.asarray([1,0,0,0])
        b = Quat(a)
        self.assertTrue((b.numpy() == a).all())


    def test_init_from_values(self):
        a = np.asarray([1,0,0,0])
        b = Quat(a[0], a[1], a[2], a[3])
        self.assertTrue((b.numpy() == a).all())


    def test_repr(self):
        a = np.random.rand(4) - 0.5
        b = Quat(a)
        self.assertEqual(b.__repr__(), f"Quat({a[0]}, {a[1]}, {a[2]}, {a[3]})")


    def test_str(self):
        a = np.random.rand(4) - 0.5
        b = Quat(a)
        self.assertEqual(str(b), f"Quaternion {a[0]} + {a[1]}i + {a[2]}j + {a[3]}k")

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
        print("\n\n!!!Unit test for static Hamilton Product not implemented!!!\n\n")
        pass


    def test_operator_matmul(self):
        


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


if __name__ == '__main__':
    unittest.main()