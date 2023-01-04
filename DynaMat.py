#!/usr/bin/env python3
"""Dynamical Matrix inverse module.

@author: hugo_prodhomme

References:
-----------
  - P. Sankowski,
"Dynamic transitive closure via dynamic matrix inverse: extended abstract"
45th Annual IEEE Symposium on Foundations of Computer Science, 2004, pp. 509-517
doi: 10.1109/FOCS.2004.25.
  - J. van den Brand, D. Nanongkai and T. Saranurak,
"Dynamic Matrix Inverse: Improved Algorithms and Matching Conditional Lower Bounds"
2019 IEEE 60th Annual Symposium on Foundations of Computer Science (FOCS), 2019, pp. 456-480,
doi: 10.1109/FOCS.2019.00036.
https://arxiv.org/pdf/1905.05067.pdf
  -  Kenneth S. Miller
"On the Inverse of the Sum of Matrices"
1981 Mathematics Magazine, 54:2, 67-72,
doi: 10.1080/0025570X.1981.11976898 
https://www.tandfonline.com/doi/abs/10.1080/0025570X.1981.11976898
"""

import numpy as np

try:
    import numexpr as _ne
    _ne.set_num_threads(_ne.detect_number_of_cores())
    _numexpr_enabled = True
except ModuleNotFoundError:
    _numexpr_enabled = False

class _DynaMatInv_Parent:
    
    def __init__(self, A, check_every):
        self.set_mat(A)
        self.check_every = check_every
        self.stderr = []
    
    def _check_counter(self):
        """Call reset every given number of transformations."""
        self.transforms_counter += 1
        if self.check_every:
            if self.transforms_counter % self.check_every == 0:
                self.reset_consistency()
    
    def set_mat(self, A):
        assert A.shape[-1] == A.shape[-2]
        self._A = A.copy()  # Avoid modifying input array
        self._Ainv = np.linalg.inv(A)
        self.transforms_counter = 0
        self._N = A.shape[-1]
        if A.ndim == 3:
            self._N_mat = A.shape[0]
        else:
            self._N_mat = 1
    
    def get_mat(self):
        """Return the transformed (current) matrix."""
        return self._A

    def get_inv(self):
        """Return the dynamic (current) inverse of matrix."""
        return self._Ainv

    def reset_consistency(self):
        """Recompute and replace dynamic inverse with numpy, store std_err."""
        invA = np.linalg.inv(self._A)
        self.stderr.append(np.std(self._Ainv-invA))
        self._Ainv = invA



class DynaMatInv(_DynaMatInv_Parent):
    """
    Examples
    --------
    >>> import numpy as np; rand = np.random.rand; ri = np.random.randint
    >>> from DynaMat import DynaMatInv
    >>> N = 100   
    >>> A = rand(N,N)
    >>> D = DynaMatInv(A)
    >>> D.set_col(rand(N), ri(N))
    >>> D.set_row(rand(N), ri(N))
    >>> D.set_diag_elmt(rand(), ri(N))
    >>> np.allclose(D.get_inv(), np.linalg.inv(D.get_mat()))
    """

    def __init__(self, A, check_every=24):
        """Create a matrix dynamically maintaining its inverse.
        
        Parameters
        ----------
        A : 2-d array
            Initial matrix
        reset_every : int, optional
            Inverse matrix is classically computed and updated after this
            number of column or row changes. The default is 24.
        
        Methods
        -------
        set_col(V, i) : Set the vector V as the new column at index i.
        set_row(V, i) : Set the vector V as the new row at index i.
        set_diag_elmt(v, i) : Set the i-th diagonal value to v.
        get_mat() : Get the current matrix.
        get_inv() : Get the current inverted matrix.
        """
        assert A.ndim == 2
        assert A.shape[0] == A.shape[1]
        super().__init__(A, check_every)

    def _matmul_MIcol_M(self, V, i, M):
        """Return the product of a composite-Id-column matrix with a matrix M.

        Parameters
        ----------
        V : (N)-sized array
            Column vector
        i : int
            Index of column
        M : (N,N)-sized array
            Matrix

        .. example:: Composite Id-column matrix of a vector V of size 4 
        and i=1 (second index), matrix-multplied with a (4,4) matrix M :

            [1  v0  0  0]               [m00 m01 m02 m03]
            [0  v1  0  0]               [m10 m11 m12 m13]
            [0  v2  1  0]   _matmul_    [m20 m21 m22 m33]
            [0  v3  0  1]               [m30 m31 m32 m33]
            
                    [m00+v0*m10 m01+v0*m11  m02+v0*m12  m03+v0*m13]
                    [v1*m10     v1*m11      v1*m12      v1*m13    ]
                =   [m20+v2*m10 m21+v2*m11  m22+v2*m12  m03+v0*m13]
                    [m30+v3*m10 m31+v3*m11  m32+v3*m12  m33+v3*m13]
                    
                    { vi*miC,       on row i,                }
                =   { mRC + vR*miC, on other rows,           }
                    {     where R, C are row and col indexes.}
        """
        MiC = M[i]
        VR = V.reshape(V.size, 1)
        if _numexpr_enabled:
            res = _ne.evaluate("M+VR*MiC")
        else:
            res = M + VR*MiC
        res[i] = MiC*V[i]
        return res
    
    def _inv_MIcol(self, V, i):
        """Return the column of the inverse of a composite-Id-column matrix.
        
        .. example:: Composite Id-column matrix for a vector V of size 4 
        and i=1 (second index), inverted :

                [1  v0  0  0]       [1  -v0/v1  0  0]
                [0  v1  0  0]       [0  1/v1    0  0]        
        inv     [0  v2  1  0]   =   [0  -v2/v1  1  0]
                [0  v3  0  1]       [0  -v3/v1  0  1]
                
                                    { 1/vi  , on row i,            }
                                =   { -vR/vi, on other rows,       }
                                    {     where R is the row index.}
        """
        e = 1/V[i]
        Vinv = -V*e
        Vinv[i] = e
        return Vinv

    def set_col(self, newcol, i):
        """Set a column of the matrix and update its dynamic inverse matrix.

        Parameters
        ----------
        newcol : 1-d array
            New values of the column.
        i : int
            Index of the column.
        """
        # Update transformed Matrix
        self._A[:,i] = newcol
        
        # Evaluate transformer column
        Tc = self._Ainv @ newcol

        # # Evaluate inverse transformer column
        Tinvc = self._inv_MIcol(Tc, i)

        # Update transformed Matrix inverse
        self._Ainv = self._matmul_MIcol_M(Tinvc, i, self._Ainv)
        self._check_counter()

    def set_row(self, newrow, i):
        """Set a row of the matrix and update its dynamic inverse matrix.

        Parameters
        ----------
        newrow : 1-d array
            New values of the row.
        i : int
            Index of the row.
        """
        # Update transformed Matrix
        self._A[i,:] = newrow
        
        # Evaluate transposed transformer column
        Tct = self._Ainv.transpose() @ newrow
        
        # Evaluate inverse transposed transformer column
        Tinvct = self._inv_MIcol(Tct, i)
        
        # Update transformed Matrix inverse
        self._Ainv = self._matmul_MIcol_M(Tinvct, i, self._Ainv.transpose()).transpose()
        self._check_counter()
    
    def set_diag_elmt(self, newdelmt, i):
        """Set a diagonal element of the matrix and update its dynamix inverse.

        Parameters
        ----------
        newdelmt : [int, float, complex]
            New diagonal value
        i : int
            Diagonal index
        """
        # =============================================================================
        # Miller formula, B is rank one :
        # (A+B)^-1 = A^-1 - 1/(1+tr(B @ A^-1)) * A^-1 @ B @ A^-1,
        #        with tr() is trace(), @ is matmul
        # 
        # B is diagonal matrix of rank one => contains only one element on diagonal (Bii) =>
        # A^-1 @ B @ A^-1 = Bii * Col_i(A^-1) _tensordot_ Row_i(A^-1)
        # =============================================================================
        
        A = self._A
        Ainv = self._Ainv
        N = A.shape[0]

        Bii = newdelmt - np.diag(A)[i]
        newAinv = Ainv - 1/(1+Bii*Ainv[i,i]) * Bii * Ainv[:,i].reshape((N,1)) * Ainv[i,:].reshape((1,N))

        self._A[i,i] = newdelmt
        self._Ainv = newAinv
        self._check_counter()


class DynaMatInvStacked(_DynaMatInv_Parent):
    """
    Examples
    --------
    >>> import numpy as np; rand = np.random.rand; ri = np.random.randint
    >>> from DynaMat import DynaMatInvStacked
    >>> N = 100
    >>> N_mat = 66
    >>> A = rand(N_mat, N, N)
    >>> D = DynaMatInvStacked(A)
    >>> D.set_cols(rand(N_mat, N), ri(N))
    >>> D.set_rows(rand(N_mat, N), ri(N))
    >>> D.set_diag_elmts(rand(N_mat), ri(N))
    >>> np.allclose(D.get_inv(), np.linalg.inv(D.get_mat()))
    """
    def __init__(self, A, check_every=24):
        """Maintain the inverses of a matrices undergoing transforms.

        Parameters
        ----------
        A : 3-d array
            Initial matrices, stacked along the axis 0.
        reset_every : int, optional
            Inverse matrices are classically computed and updated after this
            number of columns or rows changes. The default is 24.
        
        Methods
        -------
            set_cols(V, i) : Set the vectors V as the new columns at index i for each matrix.
            set_rows(V, i) : Set the vectors V as the new row at index i for each matrix..
            set_diag_elmts(v, i) : Set the i-th diagonal values for each matrix to v values.
            get_mat() : Get the current matrices.
            get_inv() : Get the current inverted matrices.
        """
        assert A.ndim == 3
        assert A.shape[1] == A.shape[2]
        super().__init__(A, check_every)
        self._N_mat = A.shape[0]
        self._N = A.shape[1]

    def _matmul_MIcol_M(self, V, i, M):
        """Return the products composite-Id-column matrices with a matrices M."""
        """(Reminder):
                        { vi*miC,       on row i                }
                        { mRC + vR*miC, on other rows           }
                        {     where R, C are row and col number.}
        """
        N_mat = self._N_mat
        N = self._N
        assert V.shape == (N_mat, N, 1)
        assert M.shape == (N_mat, N, N)

        MiC = M[:,i,:].reshape(N_mat, 1, N)
        VR = V.reshape(N_mat, N, 1)
        Vi = V[:,i].reshape(N_mat, 1, 1)

        if _numexpr_enabled:
            res = _ne.evaluate("M+VR*MiC")
        else:
            res = M + VR*MiC
        res[:,i] = (MiC*Vi).reshape(N_mat, N)

        return res

    def _inv_MIcol(self, V, i):
        """Return the columns of the inverses composite-Id-column matrices."""
        """(Reminder):
                        { 1/vi  , on row i,            }
                        { -vR/vi, on other rows,       }
                        {     where R is the row index.}
        """
        N_mat = self._N_mat
        assert V.shape[0] == N_mat
        assert V.shape[2] == 1

        e = 1/V[:,i].reshape(N_mat, 1, 1)
        Vinv = -V*e
        Vinv[:,i,0] = e.reshape(N_mat)
        return Vinv

    def set_cols(self, newcols, i):
        """Set columns and re-evaluate inverses.
        Parameters
        ----------
        newcols : (N_mat,N)-size array
            New columns
        i : int
            Columns index
        """
        N = self._N
        N_mat = self._N_mat
        assert newcols.shape == (N_mat, N)
        
        # Update transformed Matrix
        self._A[:,:,i] = newcols
        
        # Evaluate transformer column
        Tc = np.matmul(self._Ainv, newcols.reshape(N_mat, N, 1))

        # Evaluate inverse transformer column
        Tinvc = self._inv_MIcol(Tc, i)

        # Update transformed Matrix inverse
        self._Ainv = self._matmul_MIcol_M(Tinvc, i, self._Ainv)
        self._check_counter()

    def set_rows(self, newrows, i):
        """Set rows and re-evaluate inverses.
        
        Parameters
        ----------
        newrows : (N_mat,N)-size array
            New rows.
        i : int
            Rows index.
        """
        N_mat = self._N_mat
        N = self._N
        assert newrows.shape == (N_mat, N)
        
        # Update transformed Matrix
        self._A[:,i,:] = newrows
        
        # Evaluate transposed transformer column
        Tct = np.matmul(self._Ainv.transpose((0,2,1)), newrows.reshape(N_mat, N, 1))
        
        # Evaluate inverse transposed transformer column
        Tinvct = self._inv_MIcol(Tct, i)
        
        # Update transformed Matrix inverse
        self._Ainv = self._matmul_MIcol_M(Tinvct, i, self._Ainv.transpose((0,2,1))).transpose((0,2,1))
        self._check_counter()
    
    def set_diag_elmts(self, newdelmts, i):
        """Set a diagonal element of the matrix and update its dynamix inverse.

        Parameters
        ----------
        newdelmts : [int, float, complex]
            New diagonal values
        i : int
            Diagonal index
        """
        # =============================================================================
        # Miller formula, B is rank one :
        # (A+B)^-1 = A^-1 - 1/(1+tr(B @ A^-1)) * A^-1 @ B @ A^-1,
        #        with tr() is trace(), @ is matmul
        # 
        # B is diagonal matrix of rank one => contains only one element on diagonal (Bii) =>
        # A^-1 @ B @ A^-1 = Bii * Col_i(A^-1) _tensordot_ Row_i(A^-1)
        # =============================================================================
        assert newdelmts.shape == (self._N_mat, )
        
        A = self._A
        Ainv = self._Ainv
        N = self._N
        N_mat = self._N_mat

        Bii = newdelmts - A[:, i, i]
        fac = (1/(1+Bii*Ainv[:,i,i]) * Bii).reshape((N_mat, 1, 1))
        
        Ainv_col_i = Ainv[:,:,i].reshape((N_mat,N,1))
        Ainv_row_i = Ainv[:,i,:].reshape((N_mat,1,N))
        
        # newAinv = _ne.evaluate("Ainv - fac * Ainv_col_i * Ainv_row_i")
        newAinv = Ainv - fac * Ainv_col_i * Ainv_row_i
        
        self._A[:,i,i] = newdelmts
        self._Ainv = newAinv
        self._check_counter()
