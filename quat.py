# Copyright (c) 2025 Julien Fischer.
# All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.

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
        self.to_rot_vec = self._instance_to_rot_vec


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


    def conjugate_by(self, q: Union[Quat, np.ndarray]) -> Union[Quat, np.ndarray]:
        """
            Conjugates quaternion q to this quaternion, the result is given by
                q @ self @ q^-1

        Params
        ------
            q (Quat or np.ndarray):
                Quaternion that should be conjugated to this quaternion. Can be
                of shape (4), (1, 4), or (N, 4) for a batch of quaternions.

        Returns
        -------
            Quat or np.ndarray of Quat:
                This quaternion conjugated by q. If q was batched, returns a
                numpy array of Quats.
        """
        qp = q @ self
        res = Quat.hamilton_product(qp, Quat.reciprocal(q))
        if res.ndim == 1:
            return Quat(res)
        return np.asarray([Quat(elem) for elem in res])


    def conjugate_to(self, p: Union[Quat, np.ndarray]) -> Union[Quat, np.ndarray]:
        """
            Conjugates this quaternion to the given quaternion p, resulting in
                self @ p @ self^-1

        Params
        ------
            p (Quat or np.ndarray):
            Quaternion to which this quaternion should be conjugated to. Can be
            of shape (4), (1, 4), or (N, 4) for a batch of quaternions.

        Returns
        -------
            Quat or np.ndarray of Quat:
                This quaternion conjugated to p. If p was batched, returns a numpy
                array of Quats.
        """
        qp = self @ p
        res = Quat.hamilton_product(qp, self.reciprocal())
        if res.ndim == 1:
            return Quat(res)
        return np.asarray([Quat(elem) for elem in res])


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


    def _instance_to_rot_vec(self) -> np.ndarray:
        """
            Converts this quaternion to a rotation vector

        Returns
        -------
            np.ndarray:
                The rotation vector that describes the rotation of this normed quaternion.
                Shape (3)
        """
        q = self.normalize()
        angle = 2* np.arccos(q.scalar())
        denominator = np.sqrt(1 - q.scalar() * q.scalar())
        if np.isclose(denominator, 0):
            raise ZeroDivisionError("Denominator is zero!")
        vec = q.vector() / denominator
        return vec * angle


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


    @staticmethod
    def conjugate_p_by_q(q: Union[Quat, np.ndarray], p: Union[Quat, np.ndarray]) -> np.ndarray:
        """
            Conjugates quaternion p by q. The result is the conjugation
                q @ p @ q^-1
            Quaternions q and p can be batched (shapes (N, 4) and (4), (4) and (N, 4), or
            (N, 4) and (N, 4) are allowed). (4) can also be represented by (1, 4).
            If p is a vector quaternion representing a position P, and q is a unit quaternion
            representing a rotation R, the result is a vector quaternion representing the new
            position R @ P when point P is rotated by R.

        Params
        ------
            q (Quat or np.ndarray):
                Quaternion(s) q that should be conjugated to p. Can be of shape (4), (1, 4), or (N, 4).
            p (Quat or np.ndarray):
                Quaternion(s) p which should be conjugated by q. Can be of shape (4), (1, 4), or (N, 4).

        Returns
        -------
            np.ndarray:
                The result of the conjugation, i.e. q @ p @ q^-1
        """
        if (not isinstance(q, Quat) and not isinstance(q, np.ndarray)) or (not isinstance(p, Quat) and not isinstance(p, np.ndarray)):
            raise TypeError(f"Expected q and p to be of types Quat or np.ndarray but got types {type(q)} and {type(p)}")

        q, q_single, p, p_single = Quat._convert_and_align(q, p)
        q_reciprocal = Quat.reciprocal(q)
        qp = Quat.hamilton_product(q, p)
        res = Quat.hamilton_product(qp, q_reciprocal)
        if q_single and p_single:
            return res[0]
        return res


    @staticmethod
    def from_point(p: np.ndarray) -> Union[Quat, np.ndarray]:
        """
            Returns the representation of point p = (x, y, z) as vector
            quaternion p' with
                p' = 0 + xi + yj + zk

        Params
        ------
            p (np.ndarray):
                Point which should be represented as vector quaternion.
                Can be of shape (3) or (N, 3)
        """
        if not isinstance(p, np.ndarray):
            raise TypeError(f"Expected argument p to be of type np.ndarray but got type {type(p)}")
        if p.ndim == 1 and len(p) == 3:
            return Quat(0, p[0], p[1], p[2])
        elif p.ndim == 2 and p.shape[1] == 3:
            return np.asarray([Quat(0, elem[0], elem[1], elem[2]) for elem in p])
        else:
            raise ValueError(f"Expected argument p to be of shape (3) or (N, 3) but got shape {p.shape}")


    @staticmethod
    def from_rot_vec(rot_vec: np.ndarray,
                     return_raw_numpy: bool = False
                     ) -> Union[Quat, np.ndarray]:
        """
            Calculates a rotation quaternion (i.e. versor, unit quaternion) from the given rotation vector.
            The rotation angle theta (in radians) is determined from the length of the rotation
            vector, while the rotation axis (x,y,z) is given by the normed rotation vector.
            Uses quaternion polar decomposition (via Taylor expansion of the exponential
            function) to determine rotation quaternion:
            q = exp(theta * rot_axis) [where rot_axis is vector quaternion representing the
                                    rotation axis]
            q = (cos(theta/2) + x * sin(theta/2)i + y * sin(theta/2)j + z * sin(theta/2)k)

        Params
        ------
            rot_vec (np.ndarray):
                Rotation vector(s) from which the quaternion(s) should be created.
                Can be of shape (3) or (N, 3)
            return_raw_numpy (bool):
                Whether a raw numpy array (potentially of shape (B, 4)) should
                be returned. Defaults to False.

        Returns
        -------
            Quat or np.ndarray of Quats:
                The calculated rotation quaternions. If rot_vec was batched, returns
                an array of Quats.
        """
        single = False
        if rot_vec.ndim == 1:
            single = True
            rot_vec = rot_vec.reshape(1, -1)
        if rot_vec.shape[1] != 3:
            raise ValueError(f"Expected rot_vec to have shape (3) or (N, 3) but got shape {rot_vec.shape}")
        angles = np.linalg.norm(rot_vec, axis=1).reshape(-1, 1)
        if np.isclose(angles, 0).any():
            indices = np.where(np.isclose(angles, 0))[0]
            angles[indices] = 1
            euler_axes = rot_vec / angles
            euler_axes[indices] = np.asarray([1,0,0])
            angles[indices] = 0
            #raise ZeroDivisionError("Rotation vector has length 0!")
        else:
            euler_axes = rot_vec / angles
        angles = angles.reshape(-1, 1)
        q = np.concatenate((np.cos(angles/2),
                            euler_axes * np.sin(angles/2)
                            ), axis=1, dtype=np.float64)
        if single:
            return Quat(q[0])
        if return_raw_numpy:
            return q
        else:
            return np.asarray([Quat(elem) for elem in q])


    @staticmethod
    def from_axis_and_angle(axis: np.ndarray, angle: np.ndarray) -> Union[Quat, np.ndarray]:
        """
            Calculates a rotation quaternion (i.e. versor) from given rotation axis and
            rotation angle (in radians). See `Quat.from_rot_vec` for details

        Params
        ------
            axis (np.ndarray):
                Euler axis. Can be of shape (3) or (N, 3)
            angle (np.ndarray):
                Rotation angles in radians. Can be of shape (1) or (N, 1)

        Returns
        -------
            Quat or np.ndarray of Quats:
                The rotation quaternions. If axis or angle was batched, returns
                an array of Quats.
        """
        axis_single = False
        angle_single = False
        if axis.ndim == 1:
            axis_single = True
            axis = axis.reshape(1, -1)
        if isinstance(angle, int) or isinstance(angle, float):
            angle = np.asarray([angle])
        if angle.ndim == 1:
            angle_single = True
            angle = angle.reshape(1, -1)
        if axis.shape[1] != 3:
            raise ValueError(f"Expected axis to have shape (3) or (N, 3) but got shape {axis.shape}")
        if angle.shape[1] != 1:
            raise ValueError(f"Expected angle to have shape (1) or (N, 1) but got shape {angle.shape}")
        if axis.shape[0] == 1 and angle.shape[0] > 1:
            axis = np.tile(axis, (angle.shape[0], 1))
        if angle.shape[0] == 1 and axis.shape[0] > 1:
            angle = np.tile(angle, (axis.shape[0], 1))
        assert axis.shape[0] == angle.shape[0], f"Axes and angles have to have the same batch size, got {axis.shape[0]} and {angle.shape[0]}"

        norms = np.linalg.norm(axis, axis=1).reshape(-1 ,1)
        if np.isclose(norms, 0).any():
            raise ZeroDivisionError("One of the given axes has length zero!")
        axis = axis / norms
        q = np.concatenate((np.cos(angle/2),
                            axis * np.sin(angle/2)
                            ), axis=1, dtype=np.float64)
        if axis_single and angle_single:
            return Quat(q[0])
        return np.asarray([Quat(elem) for elem in q])


    @staticmethod
    def rotate_point(p: Union[Quat, np.ndarray], q: Union[Quat, np.ndarray]) -> Union[Quat, np.ndarray]:
        """
            Rotates point p by quaternion q.

        Params
        ------
            p (Quat or np.ndarray):
                Point p that should be rotated. Can be an actual point (shape (3)
                or (N, 3)) or a vector quaternion that represents point p (shape (4) or (N, 4))
            q (Quat or np.ndarray):
                Rotation quaternion. Can be of shape (4) or (N, 4).
        """
        if isinstance(p, np.ndarray):
            if (p.ndim == 1 and len(p) == 3) or (p.ndim == 2 and p.shape[1] == 3):
                if p.ndim == 1 and isinstance(p[0], Quat):
                    pass
                else:
                    p = Quat.from_point(p)
        res = Quat.conjugate_p_by_q(q, p)
        if res.ndim == 1:
            return Quat(res)
        return np.asarray([Quat(elem) for elem in res])


    @staticmethod
    def rotate(point: np.ndarray, axis: np.ndarray, angle: Union[int, float, np.ndarray]) -> np.ndarray:
        """
            Uses quaternions to rotate the given point along the given axis by the
            given angle (in radians).

        Params
        ------
            point (np.ndarray):
                Point that should be rotated, can be of shape (3) or (N, 3)
            axis (np.ndarray):
                The axis the given point should be rotated around. Can be of
                shape (3) or (N, 3).
            angle (np.ndarray):
                The angle in radians by which the point should be rotated.
                Can be of shape (1) or (N, 1)
        """
        q = Quat.from_axis_and_angle(axis, angle)
        p = Quat.from_point(point)
        p_rot = Quat.rotate_point(p, q)
        if isinstance(p_rot, np.ndarray):
            return np.asarray([elem.vector() for elem in p_rot]).reshape(-1, 3)
        if isinstance(p_rot, Quat):
            return p_rot.vector()
        raise ValueError("Quat.rotate_point() returned invalid type")


    @staticmethod
    def generate_random_rotations(num_samples: int,
                                  return_float_array: bool,
                                  np_rng: np.random.Generator = None,
                                  np_random_seed = None,
                                  ) -> np.ndarray:
        """
            Generates random rotation quaternions.

        Params
        ------
            num_samples (int):
                Number of random rotation quaternions that should be generated
            return_pure_np_array (bool):
                Whether the returned array should consist of np.float64 values that
                represent the generated quaternions or consist of Quat objects
            np_rng (np.random.Generator):
                Numpy random generator that should be used. If None, uses default_rng
            np_random_seed:
                Random seed to use if a default_rng generator is created.

        Returns
        -------
            np.ndarray:
                The generated quaternions. If return_float_array is True, the
                array will be of shape (N, 4) and dtype np.float64. If
                return_float_array is False, the array will be of shape (N) and
                consist of Quat objects.
        """
        if np_rng is None:
            np_rng = np.random.default_rng(seed=np_random_seed)
        q = np_rng.standard_normal(num_samples*4).reshape(-1, 4)
        quats = Quat.normalize(q)
        if return_float_array:
            return quats
        return np.asarray([Quat(elem) for elem in quats])


    @staticmethod
    def to_rot_vec(q: Union[Quat, np.ndarray]) -> np.ndarray:
        """
            Converts the given quaternion to rotation vectors.
            The quaternions do not need to be normalized.
            This is the inverse of Quat.from_rot_vec(), although the resulting
            rotation vectors may not have the same length if the original
            vector had norm > 2 pi.

        Params
        ------
            q (Quat or np.ndarray):
                The quaternions that should be converted to rotation vectors.
                Can be of shape (4) or (N, 4)

        Returns
        -------
            np.ndarray:
                The resulting rotation vectors of shape (3) or (N, 3)
        """
        if isinstance(q, Quat):
            q = q.numpy()
        if isinstance(q, np.ndarray):
            if q.ndim == 1 and isinstance(q[0], Quat):
                q = Quat.quat_array_to_float_array(q)
            q_single = False
            if q.ndim == 1:
                q_single = True
                q = q.reshape(1, -1)
            if q.shape[1] != 4:
                raise ValueError(f"Expected argument q to have shape (3) or (N, 3) but got shape {q.shape}")
            q = Quat.normalize(q)
            scalars = q[..., 0]
            vectors = q[..., 1:]
            angles = 2* np.arccos(scalars)
            denominators = np.sqrt(1 - scalars * scalars)
            if np.isclose(denominators, 0).any():
                raise ZeroDivisionError("At least one of the denominators is zero!")
            vec = vectors / denominators.reshape(-1, 1)
            res = vec * angles.reshape(-1, 1)
            if q_single:
                return res[0]
            return res
        raise TypeError(f"Expected argument q to be of type Quat or np.ndarray but got type {type(q)}")


    @staticmethod
    def quat_array_to_float_array(arr: np.ndarray) -> np.ndarray:
        """
            Converts a np.ndarray of Quat elements to a np.ndarray of np.float64 elements

        Params
        ------
            arr (np.ndarray):
                Numpy array of Quat objects, shape (N)

        Returns
        -------
            np.ndarray:
                Numpy array with float64 elements that represent quaternions, shape (N, 4)
        """
        if arr.ndim != 1:
            arr = arr.flatten()
        if not isinstance(arr[0], Quat):
            raise ValueError(f"Expected arr to consist of objects from type Quat but got type {type(arr[0])}")
        arr_single = False
        if len(arr) == 1:
            arr_single = True
        res = np.asarray([elem.numpy() for elem in arr])
        if arr_single:
            return res[0]
        return res
