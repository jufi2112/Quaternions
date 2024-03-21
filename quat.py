"""
    TODOs

    - [] conjugate_p_by_q
    - [] rotate
    - [] create from vector representation

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
        self.hamilton_product = self._instance_hamilton_product
        self.pow = self._instance_pow
        self.reciprocal = self._instance_reciprocal
        self.div_lhand = self._instance_div_lhand
        self.div_rhand = self._instance_div_rhand


    def __str__(self):
        return f"Quaternion {self.quat_[0]} + {self.quat_[1]}i + {self.quat_[2]}j + {self.quat_[3]}k"


    def __repr__(self):
        return f"Quat({self.quat_[0]}, {self.quat_[1]}, {self.quat_[2]}, {self.quat_[3]})"


    def __neg__(self):
        return Quat(self.quat_ * -1)


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
                if ufunc is np.divide:
                    return y.__rtruediv__(x)
            if isinstance(x, np.float64) and isinstance(y, Quat):
                if ufunc is np.add:
                    return y.__radd__(x)
                if ufunc is np.subtract:
                    return y.__rsub__(x)
                if ufunc is np.multiply:
                    return y.__rmul__(x)
                if ufunc is np.matmul:
                    return y.__rmatmul__(x)
                if ufunc is np.divide:
                    return y.__rtruediv__(x)
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


    def __truediv__(self, other: Union[Quat, np.ndarray, int, float]) -> Union[Quat, np.ndarray]:
        """
            Division of q by scalar other. If you want to divide by a quaternion,
            use div_lhand() or div_rhand() methods.
        """
        if isinstance(other, int) or isinstance(other, float):
            if other == 0:
                raise ZeroDivisionError("Parameter other is zero!")
            return self * (1/other)
        if isinstance(other, Quat):
            raise TypeError("Division of one quaternion by another is not unique, use div_lhand() or div_rhand() instead.")
        if isinstance(other, np.ndarray):
            if other.ndim == 1:
                if len(other) == 1:
                    if np.isclose(other, 0).any():
                        raise ZeroDivisionError(f"Encountered zero in scalar array other: {other}")
                    return self * (1/other)
                elif len(other) == 4:
                    raise TypeError("Division of one quaternion by another is not unique, use div_lhand() or div_rhand() instead.")
                else:
                    raise ValueError(f"Expected scalar of shape (1) or (N, 1), got {other.shape}. If you want to multiply by multiple scalar values, provide them via a (N, 1)-shaped array!")
            elif other.ndim == 2:
                if other.shape[1] == 1:
                    if np.isclose(other, 0).any():
                        raise ZeroDivisionError(f"Scalar array other contains 0 entry: {other}")
                    return self * (1/other)
                elif other.shape[1] == 4:
                    raise TypeError("Division of one quaternion by another is not unique, use div_lhand() or div_rhand() instead.")
                else:
                    raise ValueError(f"Expected batched array to have shape (N, 1) or (N, 4), got {other.shape}")
        return NotImplemented


    def __rtruediv__(self, other: Union[int, float, np.ndarray]) -> Union[Quat, np.ndarray]:
        """
            Reverse true division of scalar other by this quaternion q given by:
            other * q^-1
        """
        if isinstance(other, int) or isinstance(other, float):
            return self.reciprocal() * other
        if isinstance(other, np.ndarray):
            err_msg = (f"Expected scalar of shape (1) or (N, 1), got {other.shape}. If you want " +
                        "to divide multiple scalar values by a quaternion, provide them as " +
                        "(N, 1)-shaped array!")
            if other.ndim == 1:
                if len(other) == 1:
                    return self.reciprocal() * other
                else:
                    raise ValueError(err_msg)
            elif other.ndim == 2:
                if other.shape[1] == 1:
                    return self.reciprocal() * other
                else:
                    raise ValueError(err_msg)
            else:
                raise ValueError(f"Expected scalar to be of shape (1) or (N, 1), got {other.shape}")
        return NotImplemented


    def __itruediv__(self, other: Union[int, float, np.ndarray, Quat]) -> Quat:
        if isinstance(other, int) or isinstance(other, float):
            if math.isclose(other, 0):
                raise ZeroDivisionError("Argument other is zero!")
            self = self / other
            return self
        if isinstance(other, Quat):
            raise TypeError(
                f"Division of one quaternion with another is not unique, use div_lhand_() or "
                "div_rhand_() instead"
            )
        if isinstance(other, np.ndarray):
            if other.ndim == 1 and len(other) == 1:
                self = self / other
                return self
            elif other.ndim == 2 and other.shape[0] == 1 and other.shape[1] == 1:
                other = other[0]
                self = self / other
                return self
            else:
                raise ValueError(
                    f"Expected other to be of shape (1) or (1, 1), got {other.shape}. Division of "
                    "one quaternion with another is not unique, use div_lhand_() or div_rhand_() "
                    "instead."
                )
        return NotImplemented


    def __matmul__(self, other: Union[Quat, np.ndarray]) -> Union[Quat, np.ndarray]:
        if isinstance(other, Quat):
            return Quat(Quat.hamilton_product(self, other))
        if isinstance(other, np.ndarray):
            err_msg = f"Expected other quaternion to have shape (4) or (N, 4), got {other.shape}"
            if other.ndim == 1:
                if other.shape[0] == 4:
                    return Quat(Quat.hamilton_product(self, other))
                else:
                    raise ValueError(err_msg)
            elif other.ndim == 2:
                if other.shape[1] == 4:
                    res = Quat.hamilton_product(self, other)
                    return np.asarray([Quat(elem) for elem in res])
                else:
                    raise ValueError(err_msg)
            else:
                raise ValueError(err_msg)
        return NotImplemented


    def __rmatmul__(self, other: np.ndarray) -> Union[Quat, np.ndarray]:
        if isinstance(other, np.ndarray):
            err_msg = f"Expected other quaternion to have shape (4) or (N, 4), got {other.shape}"
            if other.ndim == 1:
                if other.shape[0] == 4:
                    return Quat(Quat.hamilton_product(other, self))
                else:
                    raise ValueError(err_msg)
            elif other.ndim == 2:
                if other.shape[1] == 4:
                    res = Quat.hamilton_product(other, self)
                    return np.asarray([Quat(elem) for elem in res])
                else:
                    raise ValueError(err_msg)
            else:
                raise ValueError(err_msg)
        return NotImplemented


    def __imatmul__(self, other: Union[Quat, np.ndarray]) -> Quat:
        if isinstance(other, Quat):
            self = self @ other
            return self
        if isinstance(other, np.ndarray):
            err_msg = f"Expected other quaternion to have shape (4) or (1, 4), got {other.shape}"
            if other.ndim == 1:
                if other.shape[0] == 4:
                    self = self @ other
                    return self
                else:
                    raise ValueError(err_msg)
            elif other.ndim == 2:
                if other.shape[0] == 1 and other.shape[1] == 4:
                    self = self @ other[0]
                    return self
                else:
                    raise ValueError(err_msg)
            else:
                raise ValueError(err_msg)
        return NotImplemented


    def __pow__(self, power: int) -> Quat:
        if isinstance(power, int):
            return Quat(Quat.pow(self, power))
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
            q = 0 + 0i + 0j + 0k
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


    def copy(self) -> Quat:
        """
            Returns a copy of this quaternion.
        """
        return Quat(np.copy(self.quat_))


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

        Params
        ------
            other (int or float or np.ndarray):
                The scalar that should be multiplied to this quaternion. If multiple
                scalars should be multiplied independently to q, provide a np.ndarray
                of shape (N, 1)

        Result
        ------
            Quat or np.ndarray:
                The result s * q. If other is batched, the result is an np.ndarray
                of Quats with shape (N)
        """
        res = Quat.scalar_multiply(self, other)
        if res.ndim == 1:
            return Quat(res)
        return np.asarray([Quat(elem) for elem in res])


    def _instance_hamilton_product(self, other: Union[Quat, np.ndarray]) -> Union[Quat, np.ndarray]:
        """
            Calculates the (possibly batched) hamilton product q @ other between
            this quaternion q and the given quaternion parameter other.

        Params
        ------
            other (Quat or np.ndarray):
                The quaternion that should be multiplied to the right side of this
                quaternion. If other should be multiple quaternions that are to be
                independently multiplied to q, provide other as shape (N, 4).

        Returns
        -------
            Quat or np.ndarray:
                The result q @ other. If other is batched, the result is an
                np.ndarray of Quats with shape (N)
        """
        res = Quat.hamilton_product(self, other)
        if res.ndim == 1:
            return Quat(res)
        return np.asarray([Quat(elem) for elem in res])


    def _instance_pow(self, power: int) -> Quat:
        """
            Calculates q ^ power. Equivalent to self ** power

        Params
        ------
            power: int
                Power > 0

        Returns
        -------
            Quat:
                q ^ power
        """
        return self ** power


    def _instance_reciprocal(self) -> Quat:
        """
            Calculates the reciprocal q^-1 of this quaternion q such that
            q * q^-1 = q^-1 * q = (1 + 0i + 0j + 0k)

        Returns
        -------
            Quat:
                q^-1
        """
        if self.is_zero():
            raise ZeroDivisionError("Cannot calculate reciprocal of zero quaternion")
        return self.conjugate() * (1/(self.norm() ** 2))

    # Div is: Given m = q @ x --> x = q^-1 @ m and q = m @ x^-1
    def _instance_div_lhand(self, other: Union[int, float, Quat, np.ndarray]) -> Union[Quat, np.ndarray]:
        """
            Calculates (possibly batched) left hand division
            of this quaternion q and scalar or quaternion other:
                q / other = (other)^-1 * q
            If other is a scalar, the result is the same as div_rhand()
            and operator '/'.
            If other is a quaternion, left hand division is the inverse of the @
            operator (hamilton product) if you want to infer the quaternion to the right
            of the @ operator, i.e. given
                r = q @ p
            you can use r.div_lhand(q) to obtain
                p = q^-1 @ r.
            If you want to obtain q from the above equation, use div_rhand() instead.

        Params
        ------
            other: (int or float or Quat or np.ndarray):
                The value that should be left-hand divided to this quaternion. Can be of shape
                (1) or (N, 1) for scalars and (4) or (N, 4) for quaternions.

        Returns
        -------
            Quat or np.ndarray of Quat:
                The result of the calculation (other)^-1 * this. If other is batched, a np.ndarray
                of Quats is returned
        """
        if isinstance(other, int) or isinstance(other, float):
            return self / other
        if isinstance(other, Quat):
            return other.reciprocal() @ self
        if isinstance(other, np.ndarray):
            if (other.ndim == 1 and len(other) == 1) or (other.ndim == 2 and other.shape[1] == 1):
                return self / other
            elif other.ndim == 1 and len(other) == 4:
                return Quat(Quat.hamilton_product(Quat.reciprocal(other), self))
            elif other.ndim == 2 and other.shape[1] == 4:
                res = Quat.hamilton_product(Quat.reciprocal(other), self)
                return np.asarray([Quat(elem) for elem in res])
            else:
                raise ValueError(f"Expected other array to have shape (1), (4), (N, 1), or (N, 4) but got shape {other.shape}")
        raise TypeError(f"Expected type int, float, Quat, or np.ndarray, got {type(other)}")


    def _instance_div_rhand(self, other: Union[int, float, Quat, np.ndarray]) -> Union[Quat, np.ndarray]:
        """
            Calculates (possibly batched) right hand division
            of this quaternion q and scalar or quaternion other:
                q / other = q * (other)^-1
            If other is a scalar, the result is the same as div_lhand()
            and operator '/'.
            If other is a quaternion, right hand division is the inverse of the @
            operator (hamilton product) if you want to infer the quaternion to the left
            of the @ operator, i.e. given
                r = q @ p
            you can use r.div_rhand(p) to obtain
                q = r @ p^-1.
            If you want to obtain p from the above equation, use div_lhand() instead.

        Params
        ------
            other: (int or float or Quat or np.ndarray):
                The value that should be right-hand divided to this quaternion. Can be of shape
                (1) or (N, 1) for scalars and (4) or (N, 4) for quaternions.

        Returns
        -------
            Quat or np.ndarray of Quat:
                The result of the calculation this * (other)^-1. If other is batched, a np.ndarray
                of Quats is returned
        """
        if isinstance(other, int) or isinstance(other, float):
            return self / other
        if isinstance(other, Quat):
            return self @ other.reciprocal()
        if isinstance(other, np.ndarray):
            if (other.ndim == 1 and len(other) == 1) or (other.ndim == 2 and other.shape[1] == 1):
                return self / other
            elif other.ndim == 1 and len(other) == 4:
                return Quat(Quat.hamilton_product(self, Quat.reciprocal(other)))
            elif other.ndim == 2 and other.shape[1] == 4:
                res = Quat.hamilton_product(self, Quat.reciprocal(other))
                return np.asarray([Quat(elem) for elem in res])
            else:
                raise ValueError(f"Expected other array to have shape (1), (4), (N, 1), or (N, 4) but got shape {other.shape}")
        raise TypeError(f"Expected type int, float, Quat, or np.ndarray, got {type(other)}")


    def normalize_(self) -> Quat:
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
            raise TypeError(f"Expected parameter other to be of type Quat or np.ndarray, got {type(other)}")
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
            raise TypeError(f"Expected parameter other to be of type Quat or np.ndarray, got {type(other)}")
        self.quat_ = self.quat_ - other
        return self


    def conjugate_(self) -> Quat:
        """
            Replace this quaternion by its conjugate.
        """
        self.quat_ = Quat.conjugate(self)
        return self


    def scalar_multiply_(self, other: Union[int, float, np.ndarray]) -> Quat:
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


    def hamilton_product_(self, other: Union[Quat, np.ndarray]) -> Quat:
        """
            Replaces this quaternion q by q @ other.
        """
        if not isinstance(other, Quat) and not isinstance(other, np.ndarray):
            raise TypeError(f"Expected parameter other to be of type Quat or np.ndarray, got {type(other)}")
        if isinstance(other, np.ndarray):
            if other.ndim > 1 and other.shape[0] != 1:
                raise ValueError(f"Expected a single quaternion as argument, but got {other.shape[0]} quaternions!")
            if other.ndim == 2:
                other = other[0]
        self.quat_ = Quat.hamilton_product(self, other)
        return self


    def pow_(self, power: int) -> Quat:
        """
            Replaces this quaternion q by q ^ power
        """
        self.quat_ = Quat.pow(self, power)
        return self


    def reciprocal_(self) -> Quat:
        """
            Replaces this quaternion q by its reciprocal q^-1
        """
        if self.is_zero():
            raise ZeroDivisionError("Cannot calculate reciprocal for zero quaternion")
        self.quat_ = Quat.reciprocal(self)
        return self


    def div_lhand_(self, other: Union[int, float, Quat, np.ndarray]) -> Quat:
        """
            Inplace left-hand division of this quaternion by other:
                (other)^-1 * this

        Params
        ------
            other (int or float or Quat or np.ndarray):
                Scalar of shape (1) or (1, 1) or quaternion of shape (4) or (1, 4)

        Returns
        -------
            Quat:
                This quaternion after other has been left-hand divided
        """
        if isinstance(other, int) or isinstance(other, float):
            if math.isclose(other, 0):
                raise ZeroDivisionError("Other is zero")
            self.quat_ *= (1/other)
            return self
        if isinstance(other, Quat):
            self.quat_ = Quat.hamilton_product(other.reciprocal(), self.quat_)
            return self
        if isinstance(other, np.ndarray):
            if other.ndim == 1:
                if len(other) == 1:
                    if np.isclose(other, 0).any():
                        raise ZeroDivisionError("Other is zero!")
                    self.quat_ *= (1/other)
                    return self
                elif len(other) == 4:
                    self.quat_ = Quat.hamilton_product(Quat.reciprocal(other), self.quat_)
                    return self
            elif other.ndim == 2 and other.shape[0] == 1:
                if other.shape[1] == 1:
                    if np.isclose(other, 0).any():
                        raise ZeroDivisionError("Other is zero!")
                    self.quat_ *= (1/other[0])
                    return self
                elif other.shape[1] == 4:
                    self.quat_ = Quat.hamilton_product(Quat.reciprocal(other[0]), self.quat_)
                    return self
            raise ValueError(f"Expected other to be of shape (1), (4), (1,1), or (1,4) but got {other.shape}")
        raise TypeError(f"Expected other to be of type int, float, Quat, or np.ndarray but got {type(other)}")


    def div_rhand_(self, other: Union[int, float, Quat, np.ndarray]) -> Quat:
        """
            Inplace right-hand division of this quaternion by other:
                this * (other)^-1

        Params
        ------
            other (int or float or Quat or np.ndarray):
                Scalar of shape (1) or (1, 1) or quaternion of shape (4) or (1, 4)

        Returns
        -------
            Quat:
                This quaternion after other has been right-hand divided
        """
        if isinstance(other, int) or isinstance(other, float):
            if math.isclose(other, 0):
                raise ZeroDivisionError("Other is zero")
            self.quat_ *= (1/other)
            return self
        if isinstance(other, Quat):
            self.quat_ = Quat.hamilton_product(self.quat_, other.reciprocal())
            return self
        if isinstance(other, np.ndarray):
            if other.ndim == 1:
                if len(other) == 1:
                    if np.isclose(other, 0).any():
                        raise ZeroDivisionError("Other is zero!")
                    self.quat_ *= (1/other)
                    return self
                elif len(other) == 4:
                    self.quat_ = Quat.hamilton_product(self.quat_, Quat.reciprocal(other))
                    return self
            elif other.ndim == 2 and other.shape[0] == 1:
                if other.shape[1] == 1:
                    if np.isclose(other, 0).any():
                        raise ZeroDivisionError("Other is zero!")
                    self.quat_ *= (1/other[0])
                    return self
                elif other.shape[1] == 4:
                    self.quat_ = Quat.hamilton_product(self.quat_, Quat.reciprocal(other[0]))
                    return self
            raise ValueError(f"Expected other to be of shape (1), (4), (1,1), or (1,4) but got {other.shape}")
        raise TypeError(f"Expected other to be of type int, float, Quat, or np.ndarray but got {type(other)}")


    @staticmethod
    def _convert_and_align(q: Union[Quat, np.ndarray],
                           p: Union[Quat, np.ndarray] = None
                           ) -> Union[Tuple[np.ndarray, bool], Tuple[np.ndarray, bool, np.ndarray, bool]]:
        if isinstance(q, Quat):
            q = q.numpy()
        if isinstance(q, np.ndarray) and q.ndim == 1 and isinstance(q[0], Quat):
            q = np.asarray([x.numpy() for x in q])
        q_single = False
        if q.ndim == 1:
            q_single = True
            q = q.reshape(1, -1)

        if p is not None:
            if isinstance(p, Quat):
                p = p.numpy()
            if isinstance(p, np.ndarray) and p.ndim == 1 and isinstance(p[0], Quat):
                p = np.asarray([x.numpy() for x in p])
            p_single = False
            if p.ndim == 1:
                p_single = True
                p = p.reshape(1, -1)

            # Align arrays
            if q.shape[0] == 1 and p.shape[0] > 1:
                q = np.tile(q, (p.shape[0], 1))
            if p.shape[0] == 1 and q.shape[0] > 1:
                p = np.tile(p, (q.shape[0], 1))
            if (q.shape[1] != 4) or (p.shape[1] != 4):
                raise ValueError(f"Both quaternions have to have 4 elements, got arrays {q.shape} and {p.shape}")

            return q, q_single, p, p_single
        else:
            if q.shape[1] != 4:
                raise ValueError(f"Quaternion has to have 4 elements, got {q.shape[1]}")
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
        if not isinstance(q, Quat) and not isinstance(q, np.ndarray):
            raise TypeError(f"Expected q to be of type Quat or np.ndarray, got {type(q)}")
        if not isinstance(p, Quat) and not isinstance(p, np.ndarray):
            raise TypeError(f"Expected p to be of type Quat or np.ndarray, got {type(p)}")
        q, q_single, p, p_single = Quat._convert_and_align(q, p)
        # @see https://en.wikipedia.org/wiki/Quaternion#:~:text=of%20vector%20quaternions.-,Hamilton%20product,-%5Bedit%5D
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


    @staticmethod
    def pow(q: Union[Quat, np.ndarray], power: int) -> np.ndarray:
        """
            Calculates q^power

        Params
        ------
            q (Quat or np.ndarray):
                The quaternion for which the power should be calculated. If
                the power of multiple quaternions should be calculated, provide
                a numpy array of shape (N, 4)
            power (int):
                The power

        Returns
        -------
            np.ndarray:
                q^power. If q is batched, the returned array is also of shape (N, 4)
        """
        if not isinstance(q, Quat) and not isinstance(q, np.ndarray):
            raise TypeError(f"Expected parameter q to be of type Quat or np.ndarray, got {type(q)}")
        if not isinstance(power, int):
            raise TypeError(f"Expected parameter power to be of type int, got {type(power)}")
        if power < 0:
            raise ValueError(f"Argument power has to be non-negative")
        if power == 0:
            if isinstance(q, Quat):
                return np.asarray([1,0,0,0])
            if q.ndim == 1:
                return np.asarray([1,0,0,0])
            elif q.ndim == 2:
                return np.tile(np.asarray([1,0,0,0]), (len(q), 1))
            else:
                raise ValueError(f"Expected quaternion q to have shape (4) or (N, 4), got {q.shape}")
        res = np.copy(q) if isinstance(q, np.ndarray) else np.copy(q.numpy())
        while (power > 1):
            res = Quat.hamilton_product(res, q)
            power -= 1
        return res


    @staticmethod
    def reciprocal(q: Union[Quat, np.ndarray]) -> np.ndarray:
        """
            Calculates the multiplicative inverse (reciprocal) q^-1 of given quaternion
            q such that q * q^-1 = q^-1 * q = 1

        Params
        ------
            q (Quat or np.ndarray):
                The (possibly batched) quaternion for which the reciprocal should
                be calculated. Shape of (4) or (N, 4).

        Returns
        -------
            np.ndarray:
                q^-1 of shape (4) or (N, 4)
        """
        if not isinstance(q, Quat) and not isinstance(q, np.ndarray):
            raise TypeError(f"Expected parameter q to be of type Quat or np.ndarray, got {type(q)}")
        if isinstance(q, np.ndarray):
            if (q.ndim == 1 and len(q) != 4) or (q.ndim == 2 and q.shape[1] != 4) or (q.ndim not in [1,2]):
                raise ValueError(f"Expected quaternion to be of shape (4) or (N, 4), got {q.shape}")
        q_conj = Quat.conjugate(q)
        q_norm_squared = Quat.norm(q) ** 2
        if np.isclose(q_norm_squared, 0).any():
            raise ZeroDivisionError("Cannot calculate reciprocal of zero quaternion.")
        return q_conj * (1/q_norm_squared)


    @staticmethod
    def div_lhand(numerator: Union[Quat, np.ndarray, float, int], denominator: Union[Quat, np.ndarray, float, int]) -> np.ndarray:
        """
            For given numerator q and denominator p, calculates left hand division
                q / p = p^-1 @ q
            Does not support scalars for both numerator and denominator.

        Params
        ------
            numerator (Quat or np.ndarray or float or int):
                Scalar or quaternion numerator of shape (1), (4), (N, 1), or (N, 4)
            denominator (Quat or np.ndarray or float or int):
                Scalar or quaternion denominator of shape (1), (4), (N, 1), or (N, 4)

        Returns
        -------
            np.ndarray:
                The result (denominator)^-1 @ numerator of shape (4) or (N, 4)
        """
        # Why would you use this for scalar by scalar division?
        if (isinstance(denominator, int) or isinstance(denominator, float)) and (isinstance(numerator, int) or isinstance(numerator, float)):
            raise TypeError("Quat.div_lhand() is not designed for all scalar input")

        # Quaternion / scalar
        if isinstance(denominator, int) or isinstance(denominator, float):
            if math.isclose(denominator, 0):
                raise ZeroDivisionError("Denominator is zero!")
            return Quat.scalar_multiply(numerator, (1/denominator))

        # Scalar / Quaterion
        if isinstance(numerator, int) or isinstance(numerator, float):
            return Quat.scalar_multiply(Quat.reciprocal(denominator), numerator)

        # check if numpy arrays contain scalars
        numerator_numpy_scalar = False
        denominator_numpy_scalar = False
        if isinstance(numerator, np.ndarray):
            if (numerator.ndim == 1 and len(numerator) == 1) or (numerator.ndim == 2 and numerator.shape[1] == 1):
                numerator_numpy_scalar = True
        if isinstance(denominator, np.ndarray):
            if (denominator.ndim == 1 and len(denominator) == 1) or (denominator.ndim == 2 and denominator.shape[1] == 1):
                denominator_numpy_scalar = True
        if numerator_numpy_scalar and denominator_numpy_scalar:
            raise TypeError("Quat.div_lhand() is not designed for all scalar input")

        # Scalar / quaternion
        if numerator_numpy_scalar:
            return Quat.scalar_multiply(Quat.reciprocal(denominator), numerator)
        # Quaternion / scalar
        if denominator_numpy_scalar:
            if np.isclose(denominator, 0).any():
                raise ZeroDivisionError("Denominator contains zero scalar!")
            return Quat.scalar_multiply(numerator, (1/denominator))

        # Quaternion / quaternion
        num, num_single, denom, denom_single = Quat._convert_and_align(numerator, denominator)
        res = Quat.hamilton_product(Quat.reciprocal(denom), num)
        if num_single and denom_single:
            return res[0]
        return res


    @staticmethod
    def div_rhand(numerator: Union[Quat, np.ndarray, float, int], denominator: Union[Quat, np.ndarray, float, int]) -> np.ndarray:
        """
            For given numerator q and denominator p, calculates right hand division
                q / p = q @ p^-1
            Does not support scalars for both numerator and denominator.

        Params
        ------
            numerator (Quat or np.ndarray or float or int):
                Scalar or quaternion numerator of shape (1), (4), (N, 1), or (N, 4)
            denominator (Quat or np.ndarray or float or int):
                Scalar or quaternion denominator of shape (1), (4), (N, 1), or (N, 4)

        Returns
        -------
            np.ndarray:
                The result numerator @ (denominator)^-1 of shape (4) or (N, 4)
        """
        # Why would you use this for scalar by scalar division?
        if (isinstance(denominator, int) or isinstance(denominator, float)) and (isinstance(numerator, int) or isinstance(numerator, float)):
            raise TypeError("Quat.div_rhand() is not designed for all scalar input")

        # Quaternion / scalar
        if isinstance(denominator, int) or isinstance(denominator, float):
            if math.isclose(denominator, 0):
                raise ZeroDivisionError("Denominator is zero!")
            return Quat.scalar_multiply(numerator, (1/denominator))

        # Scalar / Quaterion
        if isinstance(numerator, int) or isinstance(numerator, float):
            return Quat.scalar_multiply(Quat.reciprocal(denominator), numerator)

        # check if numpy arrays contain scalars
        numerator_numpy_scalar = False
        denominator_numpy_scalar = False
        if isinstance(numerator, np.ndarray):
            if (numerator.ndim == 1 and len(numerator) == 1) or (numerator.ndim == 2 and numerator.shape[1] == 1):
                numerator_numpy_scalar = True
        if isinstance(denominator, np.ndarray):
            if (denominator.ndim == 1 and len(denominator) == 1) or (denominator.ndim == 2 and denominator.shape[1] == 1):
                denominator_numpy_scalar = True
        if numerator_numpy_scalar and denominator_numpy_scalar:
            raise TypeError("Quat.div_rhand() is not designed for all scalar input")

        # Scalar / quaternion
        if numerator_numpy_scalar:
            return Quat.scalar_multiply(Quat.reciprocal(denominator), numerator)
        # Quaternion / scalar
        if denominator_numpy_scalar:
            if np.isclose(denominator, 0).any():
                raise ZeroDivisionError("Denominator contains zero scalar!")
            return Quat.scalar_multiply(numerator, (1/denominator))

        # Quaternion / quaternion
        num, num_single, denom, denom_single = Quat._convert_and_align(numerator, denominator)
        res = Quat.hamilton_product(num, Quat.reciprocal(denom))
        if num_single and denom_single:
            return res[0]
        return res
