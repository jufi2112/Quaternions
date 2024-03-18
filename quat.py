"""
    TODOs

    - [] mul
    - [] conjugate_p_by_Q
    - [] rotate
    - [] create from vector representation
    - [] pow
    - [] reciprocal

"""

from __future__ import annotations
import numpy as np
from typing import Union, Tuple
import math

class Quat:
    def __init__(self,
                 a: Union[int, float, np.ndarray] = 0,
                 b: Union[int, float] = 0,
                 c: Union[int, float] = 0,
                 d: Union[int, float] = 0
                 ):
        """
            A class that represents a quaternion in the form
            q = a + bi + cj + dk
            Internally, uses numpy for calculations.

        Params
        ------
            a (int or float or np.ndarray):
                Parameter a or numpy array that contains all parameters of the quaternion. Defaults to 0
            b (int or float):
                Parameter b. Defaults to 0. Unused if a is a numpy array
            c (int or float):
                Parameter c.  Defaults to 0. Unused if a is a numpy array
            d (int or float):
                Parameter d.  Defaults to 0. Unused if a is a numpy array
        """
        if isinstance(a, np.ndarray):
            self.quat_ = a.reshape(4).astype(np.float64)
        else:
            self.quat_ = np.asarray([a, b, c, d], dtype=np.float64)
        # Overwrite static methods by member functions
        self.norm = self._instance_norm
        self.normalize = self._instance_normalize
        self.add = self._instance_add
        self.sub = self._instance_sub
        self.conjugate = self._instance_conjugate
        self.scalar_multiply = self._instance_scalar_multiply


    def __str__(self):
        return f"Quaternion {self.quat_[0]} + {self.quat_[1]}i + {self.quat_[2]}j + {self.quat_[3]}k"


    def __repr__(self):
        return f"Quat({self.quat_[0]}, {self.quat_[1]}, {self.quat_[2]}, {self.quat_[3]})"


    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if len(args) == 2 and len(kwargs) == 0:
            x = args[0]
            y = args[1]
            if isinstance(x, np.ndarray) and isinstance(y, Quat):
                if ufunc is np.add:
                    return y.__radd__(x)
                if ufunc is np.subtract:
                    return y.__rsub__(x)
                if ufunc is np.multiply:
                    return y.__rmul__(x)
                if ufunc is np.matmul:
                    return y.__rmatmul__(x)
            if isinstance(x, np.float64) and isinstance(y, Quat):
                if ufunc is np.add:
                    return y.__radd__(x)
                if ufunc is np.subtract:
                    return y.__rsub__(x)
                if ufunc is np.multiply:
                    return y.__rmul__(x)
        return NotImplemented


    def __add__(self, other: Union[Quat, np.ndarray]):
        if not isinstance(other, Quat) and not isinstance(other, np.ndarray):
            return NotImplemented
        return self.add(other)


    def __radd__(self, other: Union[Quat, np.ndarray]):
        if not isinstance(other, Quat) and not isinstance(other, np.ndarray):
            return NotImplemented
        return self.add(other)


    def __iadd__(self, other: Union[Quat, np.ndarray]) -> Quat:
        if not isinstance(other, Quat) and not isinstance(other, np.ndarray):
            return NotImplemented
        if isinstance(other, np.ndarray):
            if other.ndim > 1:
                if other.shape[0] > 1:
                    raise ValueError(f"Inplace addition only possible for single quaternion, but got {other.shape[0]}")
                else:
                    other = other[0]
        self = self.add(other)
        return self


    def __sub__(self, other: Union[Quat, np.ndarray]):
        if not isinstance(other, Quat) and not isinstance(other, np.ndarray):
            return NotImplemented
        return self.sub(other)


    def __rsub__(self, other: Union[Quat, np.ndarray]):
        if not isinstance(other, Quat) and not isinstance(other, np.ndarray):
            return NotImplemented
        res = Quat.sub(other, self)
        if res.ndim == 1:
            return Quat(res)
        return np.asarray([Quat(elem) for elem in res])


    def __isub__(self, other: Union[Quat, np.ndarray]) -> Quat:
        if not isinstance(other, Quat) and not isinstance(other, np.ndarray):
            return NotImplemented
        if isinstance(other, np.ndarray):
            if other.ndim > 1:
                if other.shape[0] > 1:
                    raise ValueError(f"Inplace subtraction only possible for single quaternion, but got {other.shape[0]}")
                else:
                    other = other[0]
        self = self.sub(other)
        return self


    def __mul__(self, other: Union[np.ndarray, float, int]) -> Union[Quat, np.ndarray]:
        """
            Scalar multiplication of quaternion q with scalar s: q*s = s*q =
            sa + sbi + scj + sdk
        """
        if isinstance(other, int) or isinstance(other, float):
            return Quat(Quat.scalar_multiply(self, other))
        if isinstance(other, Quat):
            return NotImplemented
            return Quat(Quat.hamilton_product(self, other))
        if isinstance(other, np.ndarray):
            if other.ndim == 1:
                if len(other) == 1:
                    return Quat(Quat.scalar_multiply(self, other))
                else:
                    raise ValueError(f"Expected scalar of shape (1) or (N, 1), got {other.shape}. If you want to multiply by multiple scalar values, provide them via a (N, 1)-shaped array!")
            if other.ndim == 2:
                if other.shape[1] == 1:
                    res = Quat.scalar_multiply(self, other)
                    return np.asarray([Quat(elem) for elem in res])
                else:
                    raise ValueError(f"Expected batched scalar array to have shape (N, 1), got {other.shape}")
            else:
                raise ValueError(f"Shape of provided numpy array not supported, expected (1) or (N, 1), got {other.shape}")
        return NotImplemented


    def __rmul__(self, other: Union[float, int, np.ndarray]) -> Union[Quat, np.ndarray]:
        if isinstance(other, int) or isinstance(other, float):
            return self * other
        if isinstance(other, np.ndarray):
            if other.ndim == 1:
                if len(other) == 1:
                    return self * other
                else:
                    raise ValueError(f"Expected scalar of shape (1) or (N, 1), got {other.shape}. If you want to multiply by multiple scalar values, provide them via a (N, 1)-shaped array!")
            if other.ndim == 2:
                if other.shape[1] == 1:
                    res = Quat.scalar_multiply(self, other)
                    return np.asarray([Quat(elem) for elem in res])
                else:
                    raise ValueError(f"Expected batched scalar array to have shape (N, 1), got {other.shape}")
            else:
                raise ValueError(f"Shape of provided numpy array not supported, expected (1) or (N, 1), got {other.shape}")
        return NotImplemented


    def __imul__(self, other: Union[float, int, np.ndarray]) -> Quat:
        if isinstance(other, int) or isinstance(other, float):
            self = self * other
            return self
        if isinstance(other, np.ndarray):
            if other.ndim == 1:
                self = self * other
                return self
            elif other.ndim == 2 and other.shape[0] == 1:
                self = (self * other)[0]
                return self
            else:
                raise ValueError(f"Expected scalar array to have shape (1) or (1, 1), got {other.shape}")
        return NotImplemented


    def numpy(self) -> np.ndarray:
        """
            Returns the quaternion representation q = a + bi + cj + dk
            as numpy array [a, b, c, d].

        Returns
        -------
            np.ndarray: Quaternion representation
        """
        return self.quat_


    def is_zero(self) -> bool:
        """
            Returns whether all coefficients are zero, i.e.
            q = + + 0i + 0j + 0k
        """
        return np.isclose(self.norm(), 0)


    def is_pure(self) -> bool:
        """
            Returns whether the quaternion is a pure quaternion (i.e. a=0)
        """
        return math.isclose(self.quat_[0], 0)


    def represents_point(self) -> bool:
        """
            Returns whether the quaternion represents a point in 3D
            This is equivalent to is_pure()
        """
        return self.is_pure()


    def is_versor(self) -> bool:
        """
            Returns whether the quaternion is a versor (i.e. unit quaternion).
        """
        return math.isclose(self.norm(), 1)


    def represents_rotation(self) -> bool:
        """
            Returns whether the quaternion represents a rotation in 3D.
            This is equivalent to is_versor()
        """
        return self.is_versor()


    def scalar(self) -> float:
        """
            Returns the scalar part of the quaternion
        """
        return self.quat_[0]


    def vector(self) -> np.ndarray:
        """
            Returns the vector part of the quaternion
        """
        return self.quat_[1:]


    def _instance_norm(self) -> float:
        """
            Returns the norm of this quaternion. The norm is given by
            sqrt(a^2 + b^2 + c^2 + d^2)
        """
        return np.linalg.norm(self.numpy())


    def _instance_normalize(self) -> Quat:
        """
            Normalizes this quaternion via q' = q / || q ||
        """
        if self.is_zero():
            raise ValueError(f"Cannot normalize quaternion {self} as it has zero length!")
        return Quat(self.numpy() / self.norm())


    def _instance_add(self, other: Union[Quat, np.ndarray]) -> Union[Quat, np.ndarray]:
        """
            Adds the given quaternion to this quaternion and returns the result.

        Params
        ------
            other (Quat or np.ndarray):
                The possibly batched quaternion to add to this one

        Returns
        -------
            Quat or np.ndarray:
                The result. If other is batched, returns an array of Quats
                with shape (N)
        """
        res = Quat.add(self, other)
        if res.ndim == 1:
            return Quat(res)
        else:
            return np.asarray([Quat(elem) for elem in res])


    def _instance_sub(self, other: Union[Quat, np.ndarray]) -> Union[Quat, np.ndarray]:
        """
            Subtracts the given quaternion from this quaternion and returns the result.

        Params
        ------
            other (Quat or np.ndarray):
                The possibly batched quaternion that should be subtracted from this one.

        Returns
        -------
            Quat or np.ndarray:
                The result. If other is batched, returns an array of Quats
                with shape (N)
        """
        res = Quat.sub(self, other)
        if res.ndim == 1:
            return Quat(res)
        else:
            return np.asarray([Quat(elem) for elem in res])


    def _instance_conjugate(self):
        """
            Returns the conjugate q* = a - bi - cj - dk
            of this quaternion.
        """
        return Quat(Quat.conjugate(self))


    def _instance_scalar_multiply(self, other: Union[int, float, np.ndarray]) -> Union[Quat, np.ndarray]:
        """
            Multiplies the given scalar s to this q: q * s = sa + sbi + scj + sdk
        """
        res = Quat.scalar_multiply(self, other)
        if res.ndim == 1:
            return Quat(res)
        else:
            return np.asarray([Quat(elem) for elem in res])


    def normalize_(self):
        """
            Normalizes this quaternion.
        """
        if self.is_zero():
            raise ValueError(f"Cannot normalize quaternion {self} since it has zero length!")
        self.quat_ = self.quat_ / self.norm()
        return self


    def add_(self, other: Union[Quat, np.ndarray]) -> Quat:
        """
            Adds the given quaternion to this quaternion.
        """
        if isinstance(other, np.ndarray):
            if other.ndim > 1 and other.shape[0] != 1:
                raise ValueError(f"Expected a single quaternion as argument, but got {other.shape[0]} quaternions.")
            if other.ndim == 2:
                other = other[0]
        elif isinstance(other, Quat):
            other = other.numpy()
        else:
            return NotImplemented
        self.quat_ = self.quat_ + other
        return self


    def sub_(self, other: Union[Quat, np.ndarray]) -> Quat:
        """
            Subtracts the given quaternion from this quaternion.
        """
        if isinstance(other, np.ndarray):
            if other.ndim > 1 and other.shape[0] != 1:
                raise ValueError(f"Expected a single quaternion as argument, but got {other.shape[0]} quaternions.")
            if other.ndim == 2:
                other = other[0]
        elif isinstance(other, Quat):
            other = other.numpy()
        else:
            return NotImplemented
        self.quat_ = self.quat_ - other
        return self


    def conjugate_(self):
        """
            Replace this quaternion by its conjugate.
        """
        self.quat_ = Quat.conjugate(self)
        return self


    def scalar_multiply_(self, other: Union[int, float, np.ndarray]):
        """
            Replace this quaternion q by scalar multiplication with scalar s:
            q * s = s * q = sa + sbi + scj + sdk
        """
        if isinstance(other, np.ndarray):
            if other.ndim > 1 and other.shape[0] != 1:
                raise ValueError(f"Expected a single scalar as argument, but got {other.shape[0]} scalars!")
            if other.ndim == 2:
                other = other[0]
        self.quat_ = Quat.scalar_multiply(self, other)
        return self


    @staticmethod
    def _convert_and_align(q: Union[Quat, np.ndarray],
                           p: Union[Quat, np.ndarray] = None
                           ) -> Union[Tuple[np.ndarray, bool], Tuple[np.ndarray, bool, np.ndarray, bool]]:
        if isinstance(q, Quat):
            q = q.numpy()
        q_single = False
        if q.ndim == 1:
            q_single = True
            q = q.reshape(1, -1)

        if p is not None:
            if isinstance(p, Quat):
                p = p.numpy()
            p_single = False
            if p.ndim == 1:
                p_single = True
                p = p.reshape(1, -1)

            # Align arrays
            if q.shape[0] == 1 and p.shape[0] > 1:
                q = np.tile(q, (p.shape[0], 1))
            if p.shape[0] == 1 and q.shape[0] > 1:
                p = np.tile(p, (q.shape[0], 1))
            assert q.shape[1] == p.shape[1] == 4, f"Both quaternions have to have 4 elements, got {q.shape[1]} and {p.shape[1]}"

            return q, q_single, p, p_single
        else:
            assert q.shape[1] == 4, f"Quaternion has to have 4 elements, got {q.shape[1]}"
            return q, q_single


    @staticmethod
    def norm(q: Union[Quat, np.ndarray]) -> Union[float, np.ndarray]:
        """
            Returns the norm || q || of the given quaternion q = a + bi + cj + dk
            with || q || = sqrt(a^2 + b^2 + c^2 + d^2)

        Params
        ------
            q (Quat or np.ndarray):
                The (possibly batched) quaternion(s) for which the norm should
                be calculated.

        Returns
        -------
            float or np.ndarray:
                Norm of the quaternion(s).
        """
        q, single = Quat._convert_and_align(q)
        norm = np.linalg.norm(q, axis=1).reshape(-1, 1)
        if single:
            return norm[0]
        return norm


    @staticmethod
    def normalize(q: Union[Quat, np.ndarray]) -> np.ndarray:
        """
            Normalizes (possibly batched) quaternion q
        """
        q, single = Quat._convert_and_align(q)
        q_norm = Quat.norm(q)
        if np.isclose(q_norm, 0).any():
            raise ValueError("Quaternion has zero length, cannot perform normalization!")
        q_normalized = q / q_norm
        if single:
            return q_normalized[0]
        else:
            return q_normalized


    @staticmethod
    def hamilton_product(q: Union[Quat, np.ndarray], p: Union[Quat, np.ndarray]) -> np.ndarray:
        """
            Calculates the Hamilton product of (possibly batched) quaternions q and p
            Note that in general, this is not commutative: q*p != p*q

        Params
        ------
            q (Quat or np.ndarray):
                Possibly batched left-side quaternion.
            p (Quat or np.ndarray):
                Possibly batched right-side quaternion

        Returns
        -------
            np.ndarray:
                The Hamilton product q * p
        """
        q, q_single, p, p_single = Quat._convert_and_align(q, p)
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
    def add(q: Union[Quat, np.ndarray], p: Union[Quat, np.ndarray]) -> np.ndarray:
        """
            Adds both quaternions (q+p) = a1+a2 + (b1+b2)i + (c1+c2)j + (d1+d2)k

        Params
        ------
            q (Quat or np.ndarray):
                Possibly batched left-side quaternion
            p (Quat or np.ndarray):
                Possibly batched right-side quaternion

        Returns
        -------
            np.ndarray:
                Sum q + p
        """
        q, q_single, p, p_single = Quat._convert_and_align(q, p)
        res = q + p
        if q_single and p_single:
            return res[0]
        return res

    @staticmethod
    def sub(q: Union[Quat, np.ndarray], p: Union[Quat, np.ndarray]) -> np.ndarray:
        """
            Subtracts quaternion p from quaternion q.

        Params
        ------
            q (Quat or np.ndarray):
                Quaternion from which should be subtracted
            p (Quat or np.ndarray):
                Quaternion which should be subtracted

        Returns
        -------
            np.ndarray:
                q - p
        """
        q, q_single, p, p_single = Quat._convert_and_align(q, p)
        res = q - p
        if q_single and p_single:
            return res[0]
        return res


    @staticmethod
    def conjugate(q: Union[Quat, np.ndarray]) -> np.ndarray:
        """
            Returns the conjugate q* of q = a + bi + cj + dk
            with q* = a - bi - cj - dk

        Params
        ------
            q (Quat or np.ndarray):
                The possibly batched quaternion for which the
                conjugate should be calculated

        Returns
        -------
            np.ndarray:
                The conjugate
        """
        q, single = Quat._convert_and_align(q)
        res = q * np.asarray([1, -1, -1, -1])
        if single:
            return res[0]
        return res


    @staticmethod
    def scalar_multiply(q: Union[Quat, np.ndarray], s: Union[float, int, np.ndarray]) -> np.ndarray:
        """
            Calculates (possibly batched) scalar multiplication of quaternion q with scalar s:
            s * q = q * s = sa + sbi + scj + sdk

        Params
        ------
            q (Quat or np.ndarray):
                Quaternion to which the scalar should be multiplied. If multiple quaternions are provided
                via shape (N, 4), the scalar has to be int, float, or array of shape (N, 1).
            s (float or int or np.ndarray):
                Scalar that should be multiplied to the quaternion. If you want to multiply
                multiple scalars (separately) to the quaternion, provide them as array with
                shape (N, 1).

        Returns
        -------
            np.ndarray:
                The resulting quaternion. If multiple scalars where provided,
                The result is a (N, 4) array, otherwise (4)
        """
        if not isinstance(q, Quat) and not isinstance(q, np.ndarray):
            raise TypeError(f"Expected parameter q to be of type Quat or np.ndarray, got {type(q)}")
        if not isinstance(s, int) and not isinstance(s, float) and not isinstance(s, np.ndarray):
            raise TypeError(f"Expected parameter s to be of type int, float, or np.ndarray, got {type(s)}")
        if isinstance(q, Quat):
            q = q.numpy()
        # Verify that q is a valid quaternion representation: shape (4) or (N, 4)
        if q.ndim == 1 and q.shape[0] != 4:
            raise ValueError(f"Expected single quaternion to have shape (4), got {q.shape}.")
        if q.ndim == 2 and q.shape[1] != 4:
            raise ValueError(f"Expected batched quaternions to have shape (N, 4), got {q.shape}")
        if q.ndim not in [1, 2]:
            raise ValueError(f"Expected quaternion to have shape (4) or (N, 4), got {q.shape}")

        if isinstance(s, float) or isinstance(s, int):
            # One or multiple quaternions times 1 scalar
            return q * s

        if isinstance(s, np.ndarray):
            # Verify that s has valid shape: (1), (N), or (N, 1) where N == batch size of quaternion if quaternion is batched
            # Reshape (N) to (N, 1)
            if s.ndim == 1 and s.shape[0] != 1:
                raise ValueError(f"Expected batched scalar to have shape (N, 1) but got ({s.shape})")
            if s.ndim == 2 and s.shape[1] != 1:
                raise ValueError(f"Expected batched scalar s to have shape (N, 1) but got (N, {s.shape[1]})!")
            if s.ndim > 2:
                raise ValueError(f"Expected scalar array to have 1 or 2 dimensions, got {s.ndim}")
            if q.ndim > 1 and s.ndim > 1 and s.shape[0] != 1 and q.shape[0] != s.shape[0]:
                raise ValueError(f"Cannot multiply {s.shape[0]} scalars to {q.shape[0]} quaternions.")
            return q * s
