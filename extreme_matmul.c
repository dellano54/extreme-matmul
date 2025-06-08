// matrix multiplication with mkl
// Author: Dellano Samuel Fernandez
// Date: 01/06/25

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <mkl.h>

#define TINY 32
#define SMALL 128

static void simple_matmul(int m, int n, int k,
                         const float *a, int lda,
                         const float *b, int ldb,
                         float *c, int ldc)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            // inner product calculation
            for (int p = 0; p < k; p++) {
                sum += a[i*lda + p] * b[p*ldb + j];
            }
            c[i*ldc + j] = sum;
        }
    }
}

static int matmul_core(PyArrayObject* a, PyArrayObject* b, PyArrayObject* out)
{
    // get data pointers
    float *A = (float*)PyArray_DATA(a);
    float *B = (float*)PyArray_DATA(b);
    float *C = (float*)PyArray_DATA(out);
    
    npy_intp *a_shape = PyArray_SHAPE(a);
    npy_intp *b_shape = PyArray_SHAPE(b);
    npy_intp *out_shape = PyArray_SHAPE(out);
    
    int ndim_a = PyArray_NDIM(a);
    int ndim_b = PyArray_NDIM(b);
    int ndim_out = PyArray_NDIM(out);
    
    // extract matrix dimensions for blas
    int M = out_shape[ndim_out-2];
    int N = out_shape[ndim_out-1];
    int K = (ndim_a > 1) ? a_shape[ndim_a-1] : a_shape[0];
    
    int lda = (ndim_a == 2) ? K : a_shape[2];
    int ldb = (ndim_b == 2) ? N : b_shape[2];
    int ldc = N;

    // choose algorithm based on size
    if (M <= TINY && N <= TINY && K <= TINY) {
        simple_matmul(M, N, K, A, lda, B, ldb, C, ldc);
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
    }
    
    return 1;
}

static PyObject* matmul(PyObject* self, PyObject* args) {
    PyArrayObject *a, *b;
    
    // parse input arguments
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &a, &PyArray_Type, &b)) {
        PyErr_SetString(PyExc_ValueError, "need two arrays");
        return NULL;
    }

    if (PyArray_TYPE(a) != NPY_FLOAT32 || PyArray_TYPE(b) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_ValueError, "float32 only");
        return NULL;
    }

    int ndim_a = PyArray_NDIM(a);
    int ndim_b = PyArray_NDIM(b);
    
    if (ndim_a < 1 || ndim_b < 1 || ndim_a > 3 || ndim_b > 3) {
        PyErr_SetString(PyExc_ValueError, "1D to 3D arrays only");
        return NULL;
    }

    // figure out output shape
    npy_intp *a_shape = PyArray_SHAPE(a);
    npy_intp *b_shape = PyArray_SHAPE(b);
    npy_intp out_shape[3];
    int out_ndim = 0;

    if (ndim_a == 1 && ndim_b == 1) {
        if (a_shape[0] != b_shape[0]) {
            PyErr_SetString(PyExc_ValueError, "size mismatch");
            return NULL;
        }
        out_ndim = 0;
    }
    else if (ndim_a == 1 && ndim_b == 2) {
        if (a_shape[0] != b_shape[0]) {
            PyErr_SetString(PyExc_ValueError, "size mismatch");
            return NULL;
        }
        out_ndim = 1;
        out_shape[0] = b_shape[1];
    }
    else if (ndim_a == 2 && ndim_b == 1) {
        if (a_shape[1] != b_shape[0]) {
            PyErr_SetString(PyExc_ValueError, "size mismatch");
            return NULL;
        }
        out_ndim = 1;
        out_shape[0] = a_shape[0];
    }
    else {
        int inner_a = a_shape[ndim_a-1];
        int inner_b = (ndim_b == 1) ? b_shape[0] : b_shape[ndim_b-2];
        
        if (inner_a != inner_b) {
            PyErr_SetString(PyExc_ValueError, "size mismatch");
            return NULL;
        }
        
        out_ndim = (ndim_a > ndim_b) ? ndim_a : ndim_b;
        if (ndim_a == ndim_b) {
            for (int i = 0; i < ndim_a-2; i++) {
                if (a_shape[i] != b_shape[i]) {
                    PyErr_SetString(PyExc_ValueError, "batch size mismatch");
                    return NULL;
                }
                out_shape[i] = a_shape[i];
            }
        }
        else if (ndim_a > ndim_b) {
            for (int i = 0; i < ndim_a-2; i++) {
                out_shape[i] = a_shape[i];
            }
        }
        else {
            for (int i = 0; i < ndim_b-2; i++) {
                out_shape[i] = b_shape[i];
            }
        }
        
        out_shape[out_ndim-2] = a_shape[ndim_a-2];
        out_shape[out_ndim-1] = (ndim_b == 1) ? 1 : b_shape[ndim_b-1];
    }

    // create output array
    PyArray_Descr *dtype = PyArray_DescrFromType(NPY_FLOAT32);
    PyArrayObject *out = (PyArrayObject*)PyArray_NewFromDescr(
        &PyArray_Type, dtype, out_ndim, out_shape, NULL, NULL,
        NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE, NULL);
    
    if (!out) {
        PyErr_SetString(PyExc_RuntimeError, "array creation failed");
        return NULL;
    }

    // handle dot product case
    if (out_ndim == 0) {
        float *a_data = (float*)PyArray_DATA(a);
        float *b_data = (float*)PyArray_DATA(b);
        float *out_data = (float*)PyArray_DATA(out);
        
        float sum = 0.0f;
        for (int i = 0; i < a_shape[0]; i++) {
            sum += a_data[i] * b_data[i];
        }
        *out_data = sum;
        
        return (PyObject*)out;
    }

    // make arrays contiguous for blas
    PyArrayObject *a_contig = (PyArrayObject*)PyArray_FROM_OTF(
        (PyObject*)a, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    PyArrayObject *b_contig = (PyArrayObject*)PyArray_FROM_OTF(
        (PyObject*)b, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    
    if (!a_contig || !b_contig) {
        Py_XDECREF(a_contig);
        Py_XDECREF(b_contig);
        Py_DECREF(out);
        PyErr_SetString(PyExc_RuntimeError, "contiguous array failed");
        return NULL;
    }

    if (!matmul_core(a_contig, b_contig, out)) {
        Py_DECREF(a_contig);
        Py_DECREF(b_contig);
        Py_DECREF(out);
        return NULL;
    }

    Py_DECREF(a_contig);
    Py_DECREF(b_contig);
    
    return (PyObject*)out;
}

static PyMethodDef methods[] = {
    {"matmul", matmul, METH_VARARGS, "fast matrix multiply"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "extreme_matmul",
    "matrix multiplication with mkl",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_extreme_matmul(void) {
    import_array();
    return PyModule_Create(&module);
}
