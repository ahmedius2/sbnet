#include <torch/extension.h>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

extern std::pair<torch::Tensor, torch::Tensor> ReduceMask(torch::Tensor mask,
		std::pair<int,int> bcount_dynamic,
		std::pair<int,int> bsize_dynamic,
		std::pair<int,int> bstride_dynamic,
		std::pair<int,int> boffset_dynamic,
		bool avgpool_,
		float tol_);

extern torch::Tensor SparseGather(torch::Tensor x,
		torch::Tensor bin_counts_tensor,
		torch::Tensor activeBlockIndices,
		std::pair<int,int> bsize_dynamic,
		std::pair<int,int> bstride_dynamic,
		std::pair<int,int> boffset_dynamic,
		bool transpose_);

extern torch::Tensor SparseScatter(torch::Tensor x,
		torch::Tensor bin_counts_tensor,
		torch::Tensor activeBlockIndices,
		std::vector<int64_t> out_size,
		std::pair<int,int> bsize_dynamic,
		std::pair<int,int> bstride_dynamic,
		std::pair<int,int> boffset_dynamic,
		bool transpose_,
		bool add_,
		bool atomic_);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("reduce_mask", &ReduceMask, "Mask reducing operation.",
		py::arg("mask"), py::arg("bcount"), py::arg("bsize"), py::arg("bstride"),
		py::arg("boffset"), py::arg("avgpool"), py::arg("tol"));
	m.def("sparse_gather", &SparseGather, "Gather sparse blocks.",
		py::arg("x"), py::arg("bin_counts"), py::arg("active_block_indices"),
		py::arg("bsize"), py::arg("bstride"), py::arg("boffset"), 
		py::arg("transpose"));
	m.def("sparse_scatter", &SparseScatter, "Scatter sparse blocks.",
		py::arg("x"), py::arg("bin_counts"), py::arg("active_block_indices"),
		py::arg("out_size"), py::arg("bsize"), py::arg("bstride"), py::arg("boffset"), 
		py::arg("transpose"), py::arg("add"), py::arg("atomic"));
}
