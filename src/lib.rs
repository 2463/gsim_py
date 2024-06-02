use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::PyValueError;
use gsim;
use nalgebra::{DMatrix, Matrix, DVector, ViewStorage, Dyn};
use num::complex::Complex64;
use numpy::{IntoPyArray, Ix1, PyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray};
use numpy::ndarray::{ArrayBase, Ix2, OwnedRepr};


type CMatrix = DMatrix<Complex64>;
type CMatrixView<'a> = Matrix<Complex64, Dyn, Dyn, ViewStorage<'a, Complex64, Dyn, Dyn, Dyn, Dyn>>;

// fn view_to_matrix(arr: ArrayBase<ViewRepr<&Complex64>, Ix2>)->CMatrix{
//     let (rows, cols) = arr.dim();
//     let elements = arr.iter().cloned().collect::<Vec<Complex64>>();
//     CMatrix::from_iterator(rows, cols, elements.into_iter())
// }

fn array_to_matrix(arr: ArrayBase<OwnedRepr<Complex64>, Ix2>)->CMatrix{
    let (rows, cols) = arr.dim();
    CMatrix::from_iterator(rows, cols, arr.into_iter())
}

fn list_to_vec_matrix<'py>(list: &Bound<'py,PyList>)->Result<Vec<CMatrix>,PyErr>{
    let mut result:Vec<CMatrix> = Vec::new();
    for item in list.iter(){
        let pyarr = item.extract::<&PyArray2<Complex64>>()?;
        let rust_array = unsafe {
            pyarr.as_array().to_owned()
        };
        result.push(array_to_matrix(rust_array))
    }
    Ok(result)
}

// fn reailze_view_list(vector: &Vec<CMatrixView>)->Vec<CMatrix>{
//     vector.iter()
//     .map(|elem|realize_view(elem))
//     .collect()
// }

fn realize_view(mat : &CMatrixView)->CMatrix{
    let (nrows, ncols) = (mat.nrows(),mat.ncols());
    CMatrix::from_iterator(nrows, ncols, mat.iter().cloned())
}

fn pyarrayvector_to_dvector<'py>(list: &Bound<'py,PyArray1<f64>>)->Result<DVector<f64>,PyErr>{
    let size = list.len();
    let rust_array = unsafe {
        list.as_array().to_owned()
    };
    Ok(DVector::from_iterator(size,rust_array.into_iter()))
}

fn pyarraymatrix_to_matrix<'py>(matrix: &Bound<'py,PyArray2<Complex64>>)->Result<DMatrix<Complex64>,PyErr>{
    let rust_mat = match unsafe {
        matrix.try_as_matrix()
    } {
        Some(item) => item,
        None => return Err(PyValueError::new_err("not matrix error."))
    };
    Ok(realize_view(&rust_mat))
}

fn process_py_observable<'py, T: numpy::Element>(py: Python<'py>, mat:&DMatrix<T>)->Bound<'py,PyArray<T,Ix2>>{
    let shape = (mat.nrows(), mat.ncols());
    match mat.as_slice()  
        .to_pyarray_bound(py)
        .reshape(shape){
            Ok(pyarr) => pyarr,
            Err(_e) => panic!("reshape err")
        }
}

fn process_py_e_in<'py,T: numpy::Element>(py: Python<'py>, dvector: &DVector<T>)->Bound<'py,PyArray<T,Ix1>>{
    dvector.as_slice().to_pyarray_bound(py)
}

fn process_py_tuple_into_rust_tuple<'py>(pytuplelist: &Bound<'py, PyList>)->Result<Vec<(f64,usize)>,PyErr>{
    let mut rust_tuple_vec : Vec<(f64,usize)> = Vec::new();

    for item in pytuplelist{
        let tuple = item.downcast::<PyTuple>()?;
        let param: f64 = tuple.get_item(0)?.extract::<f64>()?;
        let gate_num: usize = tuple.get_item(1)?.extract::<usize>()?;
        rust_tuple_vec.push((param,gate_num));
    }

    Ok(rust_tuple_vec)
}

fn vec_matrix_to_pylist<'py>(py: Python<'py>, vector_of_matrix: &Vec<CMatrix>)->PyResult<Bound<'py,PyList>>{
    let pylist = PyList::empty_bound(py);

    for matrix in vector_of_matrix {
        let pyarray = process_py_observable(py,&matrix);
        pylist.append(pyarray)?;
    }
    Ok(pylist)
}

#[pyfunction]
fn ndarray_to_ndarray_py<'py>(py: Python<'py>,arr: PyReadonlyArray2<Complex64>,) -> Bound<'py, PyArray2<Complex64>> {
    // input
    let arr = arr.as_array();

    // 作成された配列にデータが入っているか念のため確認
    let first = arr[[0, 0]];
    let last = arr[[arr.nrows() - 1, arr.ncols() - 1]];
    println!("arr[0][0] = {}, arr[-1][-1] = {}", first, last);

    // output
    let output = arr.to_owned();
    output.into_pyarray_bound(py)
}

#[pyfunction]
fn make_gsim<'py>(
    py: Python<'py>,
    init: PyReadonlyArray2<'py, Complex64>,
    obs: PyReadonlyArray2<'py, Complex64>,
    gate_hams: Bound<'py,PyList>
)->PyResult<Bound<'py, PyDict>>{
    let init_density_matrix: CMatrix = realize_view(&init.as_matrix());
    // let init_density_matrix = view_to_matrix(init.as_array());
    let observable = realize_view(&obs.as_matrix());
    let py_observable = process_py_observable(py, &observable);

    let gate_hamiltonians = list_to_vec_matrix(&gate_hams)?;
    let gsim = gsim::make_gsim(init_density_matrix, observable, gate_hamiltonians);
    let pydict : Bound<'py, PyDict> = PyDict::new_bound(py);
    
    let dla = gsim.get_dla();
    let py_dla = vec_matrix_to_pylist(py, &dla)?;
    pydict.set_item("dla", py_dla)?;

    let e_in = gsim.get_e_in();
    let pye_in = process_py_e_in(py, &e_in);
    pydict.set_item("e_in", pye_in)?;

    let ad_rep_gate_hams = gsim.get_ad_rep_gate_hams();
    let py_ad_rep_gate_hams = vec_matrix_to_pylist(py, &ad_rep_gate_hams)?;
    pydict.set_item("ad_gates", py_ad_rep_gate_hams)?;

    pydict.set_item("obs", py_observable)?;

    Ok(pydict)
}

#[pyfunction]
fn simulate<'py>(
    gsim_dict: Bound<'py, PyDict>,
    params_and_gate_nums : Bound<'py, PyList>
)->PyResult<f64>{
    let dla = match gsim_dict.get_item("dla")?{
        Some(item) => list_to_vec_matrix(item.downcast::<PyList>()?),
        None => return Err(PyValueError::new_err("gsim does not have value 'dla'."))
    }?;
    let e_in = match gsim_dict.get_item("e_in")?{
        Some(item) => pyarrayvector_to_dvector(item.downcast::<PyArray1<f64>>()?),
        None => return Err(PyValueError::new_err("gsim does not have value 'dla'."))
    }?;
    let ad_reped_gates = match gsim_dict.get_item("ad_gates")?{
        Some(item) => list_to_vec_matrix(item.downcast::<PyList>()?),
        None => return Err(PyValueError::new_err("gsim does not have value 'ad_gates'."))
    }?;
    let observable = match gsim_dict.get_item("obs")?{
        Some(item) => pyarraymatrix_to_matrix(item.downcast::<PyArray2<Complex64>>()?),
        None => return Err(PyValueError::new_err("gsim does not have value 'ad_gates'."))
    }?;
    let rust_params_ad_gate_nums = match process_py_tuple_into_rust_tuple(&params_and_gate_nums){
        Ok(vector)=> vector,
        Err(_e)=> return Err(PyValueError::new_err("params_and_gate_nums has wrong value. It should be [(f64, usize)]"))
    };


    Ok(gsim::simulate_with_params(
        &dla,
        &e_in,
        &ad_reped_gates,
        &observable,
        &rust_params_ad_gate_nums))
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn gsim_py(m: &Bound<'_,PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(ndarray_to_ndarray_py, m)?)?;
    m.add_function(wrap_pyfunction!(make_gsim, m)?)?;
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    Ok(())
}