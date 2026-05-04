import re
from pathlib import Path
from conan import ConanFile
from conan.tools.files import copy
from conan.tools.cmake import cmake_layout, CMake, CMakeToolchain, CMakeDeps
from conan.tools.build import check_min_cppstd


class OdeConan(ConanFile):
    name = "ode"
    settings = "os", "arch", "compiler", "build_type"
    exports_sources = "include/*", "tests/*"
    no_copy_source = True
    package_type = "header-library"
    options = {
        "asan":     [True, False],
        "coverage": [True, False],
        "perf":     [True, False],
    }
    default_options = {
        "asan":     False,
        "coverage": False,
        "perf":     False,
    }

    def set_version(self):
        with open("VERSION") as f:
            self.version = f.read().strip()

    def build_requirements(self):
        self.test_requires("catch2/3.4.0")

    def validate(self):
        check_min_cppstd(self, 20)

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["ENABLE_ASAN"]     = self.options.asan
        tc.variables["ENABLE_COVERAGE"] = self.options.coverage
        tc.variables["ENABLE_PERF"]     = self.options.perf
        tc.generate()
        CMakeDeps(self).generate()

    def build(self):
        if self.conf.get("tools.build:skip_test", default=False):
            return
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
        if not self.options.perf:
            if self.options.asan:
                self.run(
                    f"ctest --test-dir {self.build_folder} --output-on-failure",
                    ignore_errors=True
                )
            else:
                cmake.ctest(cli_args=["--output-on-failure"])

    def package(self):
        copy(self, "*.hpp", self.source_folder, self.package_folder)

    def package_info(self):
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = []

    def package_id(self):
        self.info.clear()