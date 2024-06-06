import glob
import os
import platform
import re
from packaging.version import parse as parse_version
from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

EXT_TYPE = 'pytorch'
cmd_class = {'build_ext': BuildExtension}

def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available():
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)

def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages

def get_extensions():
    extensions = []
    
    if EXT_TYPE == 'pytorch':
        ext_name = 'mmcv._ext'
        # prevent ninja from using too many resources
        try:
            import psutil
            num_cpu = len(psutil.Process().cpu_affinity())
            cpu_use = max(4, num_cpu - 1)
        except (ModuleNotFoundError, AttributeError):
            cpu_use = 4

        os.environ.setdefault('MAX_JOBS', str(cpu_use))
        define_macros = []

        extra_compile_args = {'cxx': []}
        if platform.system() != 'Windows':
            if parse_version(torch.__version__) <= parse_version('1.12.1'):
                extra_compile_args['cxx'] = ['-std=c++14']
            else:
                extra_compile_args['cxx'] = ['-std=c++17']
        else:
            if parse_version(torch.__version__) <= parse_version('1.12.1'):
                extra_compile_args['cxx'] = ['/std:c++14']
            else:
                extra_compile_args['cxx'] = ['/std:c++17']

        include_dirs = []

        if torch.cuda.is_available():
            define_macros += [('MMCV_WITH_CUDA', None)]
            cuda_args = os.getenv('MMCV_CUDA_ARGS')
            extra_compile_args['nvcc'] = [cuda_args] if cuda_args else []
            op_files = glob.glob('./mmcv/ops/csrc/pytorch/*.cpp') + \
                glob.glob('./mmcv/ops/csrc/pytorch/cpu/*.cpp') + \
                glob.glob('./mmcv/ops/csrc/pytorch/cuda/*.cu') + \
                glob.glob('./mmcv/ops/csrc/pytorch/cuda/*.cpp')
            extension = CUDAExtension
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common'))
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common/cuda'))
        else:
            print(f'Compiling {ext_name} without CUDA')
            op_files = glob.glob('./mmcv/ops/csrc/pytorch/*.cpp') + \
                glob.glob('./mmcv/ops/csrc/pytorch/cpu/*.cpp')
            extension = CppExtension
            include_dirs.append(os.path.abspath('./mmcv/ops/csrc/common'))

        if 'nvcc' in extra_compile_args and platform.system() != 'Windows':
            if parse_version(torch.__version__) <= parse_version('1.12.1'):
                extra_compile_args['nvcc'] += ['-std=c++14']
            else:
                extra_compile_args['nvcc'] += ['-std=c++17']

        ext_ops = extension(
            name=ext_name,
            sources=op_files,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args)
        extensions.append(ext_ops)

    return extensions

setup(
    name='mmcv',
    version='0.0.1',
    description='OpenMMLab Computer Vision Foundation',
    keywords='computer vision',
    packages=[
        *find_packages(include=('mmcv', "mmcv.*")), 
        *find_packages(include=('adzoo', "adzoo.*")), 
    ],
    include_package_data=True,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Utilities',
    ],
    url='https://github.com/open-mmlab/mmcv',
    author='MMCV Contributors',
    author_email='openmmlab@gmail.com',
    install_requires=parse_requirements(),
    ext_modules= get_extensions() + [
            make_cuda_ext(
                name='iou3d_cuda',
                module='mmcv.ops.iou3d_det',
                sources=[
                    'src/iou3d.cpp',
                    'src/iou3d_kernel.cu',
                ]),
            make_cuda_ext(
                name='roiaware_pool3d_ext',
                module='mmcv.ops.roiaware_pool3d',
                sources=[
                    'src/roiaware_pool3d.cpp',
                    'src/points_in_boxes_cpu.cpp',
                ],
                sources_cuda=[
                    'src/roiaware_pool3d_kernel.cu',
                    'src/points_in_boxes_cuda.cu',
                ]),
    ],
    cmdclass=cmd_class,
    zip_safe=False)
