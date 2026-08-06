#!/usr/bin/env python3
import os
import platform
import sys
from pathlib import Path
from setuptools.command.build_ext import build_ext
import setuptools
python_executable = sys.executable

def is_macos():
    return platform.system() == "Darwin"


def is_windows():
    return platform.system() == "Windows"


def is_linux():
    return platform.system() == "Linux"


def is_arm64():
    return platform.machine() in ["arm64", "aarch64"]


def is_x86():
    return platform.machine() in ["i386", "i686", "x86_64"]

def is_for_pypi():
    ans = os.environ.get("RAPIDSPEECH_IS_FOR_PYPI", None)
    return ans is not None

def read_long_description():
    with open("README.md", encoding="utf8") as f:
        readme = f.read()
    return readme

def need_split_package():
    ans = os.environ.get("RAPIDSPEECH_SPLIT_PYTHON_PACKAGE", None)
    return ans is not None

def get_binaries():
    # No binaries needed for Python wheel — all shared libraries and the
    # pybind11 module are installed by cmake into the package directory.
    # rs-asr-offline is a CLI tool, not needed for Python bindings.
    return []

def get_package_name():
    backend = os.environ.get("RS_BACKEND", "cpu").lower()
    if backend == "cuda":
        return "rapidspeech-cuda"
    elif backend == "metal":
        return "rapidspeech-metal"
    else:
        return "rapidspeech"

package_name = get_package_name()


def sync_jieba_dict():
    """Copy third_party/cppjieba/dict into the package so the wheel ships
    the ZH G2P runtime data (`rapidspeech speak` needs it; the C++ runtime
    picks it up via RS_KOKORO_JIEBA_DICT_DIR exported by the CLI)."""
    import shutil
    root = Path(__file__).parent.resolve()
    src = root / "third_party" / "cppjieba" / "dict"
    dst = root / "rapidspeech" / "data" / "cppjieba" / "dict"
    if not src.is_dir():
        return  # submodule not checked out; wheel will lack ZH G2P data
    marker = src / "jieba.dict.utf8"
    if dst.is_dir() and (dst / "jieba.dict.utf8").exists() and \
            marker.exists() and \
            (dst / "jieba.dict.utf8").stat().st_size == marker.stat().st_size:
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


sync_jieba_dict()


def get_package_data():
    """Files bundled into the wheel next to the compiled extension:
    the model registry and the TTS G2P runtime data (~30 MB)."""
    return {
        "rapidspeech": [
            "*.so", "*.pyd", "*.dylib", "*.dll",
            "bin/*",
            "server/*",
            "python/_registry.json",
            "data/kokoro_zh/*",
            "data/kokoro_en/*",
            "data/wetext/*",
            "data/cppjieba/dict/*",
            "data/cppjieba/dict/pos_dict/*",
        ]
    }

def cmake_extension(name, *args, **kwargs) -> setuptools.Extension:
    kwargs["language"] = "c++"
    sources = []
    return setuptools.Extension(name, sources, *args, **kwargs)

class BuildExtension(build_ext):
    def build_extension(self, ext: setuptools.extension.Extension):

        # build/temp.linux-x86_64-3.8
        os.makedirs(self.build_temp, exist_ok=True)

        # build/lib.linux-x86_64-3.8
        os.makedirs(self.build_lib, exist_ok=True)

        out_bin_dir = Path(self.build_lib) / "rapidspeech"
        install_dir = Path(self.build_lib).resolve() / "rapidspeech"

        rapidspeech_dir = Path(__file__).parent.resolve()

        cmake_args = os.environ.get("RAPIDSPEECH_CMAKE_ARGS", "")
        cmake_args += f" -DPYTHON_EXECUTABLE={python_executable} -DPython3_EXECUTABLE={python_executable}"

        backend = os.environ.get("RS_BACKEND", "cpu").lower()
        if backend == "cuda":
            cmake_args += " -DRS_CUDA=ON"

        make_args = os.environ.get("RAPIDSPEECH_MAKE_ARGS", "")
        system_make_args = os.environ.get("MAKEFLAGS", "")



        if cmake_args == "":
            cmake_args = "-DCMAKE_BUILD_TYPE=Release"

        extra_cmake_args = ""
        if not need_split_package():
            extra_cmake_args += f" -DCMAKE_INSTALL_PREFIX={install_dir} "
        extra_cmake_args += " -DRS_ENABLE_PYTHON=ON "
        # `rapidspeech serve` delegates the OpenAI-compatible HTTP API and
        # WebSocket streaming endpoints to the native server.
        extra_cmake_args += " -DRS_BUILD_CLI=ON "
        extra_cmake_args += " -DRS_BUILD_TESTS=OFF "


        if "PYTHON_EXECUTABLE" not in cmake_args:
            print(f"Setting PYTHON_EXECUTABLE to {sys.executable}")
            cmake_args += f" -DPYTHON_EXECUTABLE={sys.executable}"
        else:
            extra_cmake_args += " -DPYTHON_EXECUTABLE=$(which python3)"

        # putting `cmake_args` from env variable ${rapidspeech_CMAKE_ARGS} last,
        # so they can onverride the "defaults" stored in `extra_cmake_args`
        cmake_args = extra_cmake_args + cmake_args

        if is_windows():
            if not need_split_package():
                build_cmd = f"""
             cmake {cmake_args} -B {self.build_temp} -S {rapidspeech_dir}
             cmake --build {self.build_temp} --target install --config Release -- -m:2
                """
            else:
                build_cmd = f"""
             cmake {cmake_args} -B {self.build_temp} -S {rapidspeech_dir}
             cmake --build {self.build_temp} --target rapidspeech --config Release -- -m:2
                """

            print(f"build command is:\n{build_cmd}")
            ret = os.system(
                f"cmake {cmake_args} -B {self.build_temp} -S {rapidspeech_dir}"
            )
            if ret != 0:
                raise Exception("Failed to configure sherpa")

            if not need_split_package():
                ret = os.system(
                    f"cmake --build {self.build_temp} --target install --config Release -- -m:2"  # noqa
                )
            else:
                ret = os.system(
                    f"cmake --build {self.build_temp} --target rapidspeech --config Release -- -m:2"  # noqa
                )
            if ret != 0:
                raise Exception("Failed to build and install sherpa")
        else:
            if make_args == "" and system_make_args == "":
                print("for fast compilation, run:")
                print('export RAPIDSPEECH_MAKE_ARGS="-j"; python setup.py install')
                print('Setting make_args to "-j4"')
                make_args = "-j4"

            if "-G Ninja" in cmake_args:
                if not need_split_package():
                    build_cmd = f"""
                        cd {self.build_temp}
                        cmake {cmake_args} {rapidspeech_dir}
                        ninja {make_args} install
                    """
                else:
                    build_cmd = f"""
                        cd {self.build_temp}
                        cmake {cmake_args} {rapidspeech_dir}
                        ninja {make_args} rapidspeech
                    """
            else:
                if not need_split_package():
                    build_cmd = f"""
                        cd {self.build_temp}

                        cmake {cmake_args} {rapidspeech_dir}

                        make {make_args} install/strip
                    """
                else:
                    build_cmd = f"""
                        cd {self.build_temp}

                        cmake {cmake_args} {rapidspeech_dir}

                        make {make_args} rapidspeech
                    """
            print(f"build command is:\n{build_cmd}")

            ret = os.system(build_cmd)
            if ret != 0:
                raise Exception(
                    "\nBuild rapidspeech failed. Please check the error message.\n"
                    "You can ask for help by creating an issue on GitHub.\n"
                    "\nClick:\n\thttps://github.com/RapidAI/RapidSpeech.cpp/issues/new\n"  # noqa
                )

            # After cmake install, fix NEEDED entries and RPATH on Linux.
            # cmake links against build-tree library paths, which can embed
            # relative paths like "ggml/src/libggml-base.so" into ELF NEEDED
            # entries.  auditwheel cannot locate those, so we rewrite them
            # to simple filenames and set $ORIGIN RPATH.
            if is_linux() and is_for_pypi():
                import subprocess as sp
                out_dir = Path(self.build_lib) / "rapidspeech"
                so_files = list(out_dir.glob("*.so"))
                if so_files:
                    # check if patchelf is available
                    r = sp.run(["which", "patchelf"], capture_output=True)
                    if r.returncode == 0:
                        patchelf = r.stdout.decode().strip()
                        for so_file in so_files:
                            sf = str(so_file)
                            sp.run([patchelf, "--set-rpath", "$ORIGIN", sf],
                                   capture_output=True)
                            for lib in ["libggml-base", "libggml-cpu", "libggml"]:
                                sp.run([patchelf, "--replace-needed",
                                       f"ggml/src/{lib}.so", f"{lib}.so", sf],
                                       capture_output=True)
                        print(f"Fixed NEEDED entries in {len(so_files)} .so files")
                    else:
                        print("WARNING: patchelf not found, wheel may not be repairable")


        # cmake install already places all files (pybind module, shared libs,
        # headers) into build_lib/rapidspeech/ which is the correct location
        # for the Python package.  No manual copy needed.

    def copy_extensions_to_source(self):
        # The base class copies only the extension module itself (the
        # `*.cpython-*.so`) into the source tree.  For editable installs
        # the pybind extension also needs librapidspeech-core.so and the
        # ggml shared libraries sitting next to it — its RUNPATH is
        # $ORIGIN — otherwise `import rapidspeech` fails with
        # "librapidspeech-core.so: cannot open shared object file".
        super().copy_extensions_to_source()
        import shutil
        build_pkg_dir = Path(self.build_lib) / "rapidspeech"
        src_pkg_dir = Path(__file__).parent.resolve() / "rapidspeech"
        if not build_pkg_dir.is_dir() or not src_pkg_dir.is_dir():
            return
        for so in (list(build_pkg_dir.glob("*.so")) +
                   list(build_pkg_dir.glob("*.so.*")) +
                   list(build_pkg_dir.glob("*.dylib")) +
                   list(build_pkg_dir.glob("*.dll"))):
            if "cpython-" in so.name or ".pyd" in so.name:
                continue
            dst = src_pkg_dir / so.name
            if so.is_symlink():
                target = os.readlink(so)
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                os.symlink(target, dst)
            else:
                shutil.copy2(so, dst)



try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            # The package contains compiled C++ extensions (.so/.pyd),
            # so it must always be a platform-specific wheel.
            self.root_is_pure = False

except ImportError:
    bdist_wheel = None

def get_binaries_to_install():
    if need_split_package():
        return None

    # No binaries to install via data_files for Python wheel.
    # All shared libs are installed by cmake into the package directory.
    return []


setuptools.setup(
    name=package_name,
    python_requires=">=3.7",
    use_scm_version={
        "root": ".",
        "relative_to": __file__,
    },
    setup_requires=["setuptools_scm"],
    install_requires=["numpy"],
    author="lovemefan",
    author_email="lovemefan@outlook.com",
    packages=["rapidspeech", "rapidspeech.python"],
    package_data=get_package_data(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "rapidspeech = rapidspeech.python.cli:main",
        ],
    },
    data_files=(
        [
            (
                ("Scripts", get_binaries_to_install())
                if is_windows()
                else ("bin", get_binaries_to_install())
            )
        ]
        if get_binaries_to_install()
        else None
    ),
    url="https://github.com/RapidAI/RapidSpeech.cpp",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    ext_modules=[cmake_extension("rapidspeech.rapidspeech")],
    cmdclass={"build_ext": BuildExtension, "bdist_wheel": bdist_wheel},
    zip_safe=False,
    classifiers=[
        "Programming Language :: C++",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="Apache licensed, as found in the LICENSE file",
)
