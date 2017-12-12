#!/usr/bin/python
# -*- coding: utf-8 -*-
# from brokenaxes import brokenaxes
import datetime
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


known_columns = {
    'Python loop [cpu]': ['saxpy_loop.py', 'Simple Python `for` loop.'],
    'Py Numpy [cpu]': ['saxpy_numpy.py', 'Vectorized implementation with Python [Numpy](http://www.numpy.org/) array.'],
    'Py Pandas [cpu]': ['saxpy_pandas.py', 'Vectorized implementation with Python [Pandas](https://pandas.pydata.org/) dataframe.'],
    'Java loop [cpu]': ['SaxpyLoop.java', 'Plain Java loop'],
    'Julia (loop) [cpu]': ['saxpy_loop.jl', 'Plain loop in [Julia](https://julialang.org/) programming language.'],
    'Julia (vec) [cpu]': ['saxpy_array.jl', 'Vectorized implementation with array in [Julia](https://julialang.org/) programming language.'],
    'Octave [cpu]': ['saxpy.m', 'Implementation in [GNU Octave](https://www.gnu.org/software/octave/), a high-level language primarily intended for numerical computations.'],
    'R (loop) [cpu]': ['saxpy_loop.R', 'Simple loop in [R](https://www.r-project.org/), a free software environment for statistical computing and graphics.'],
    'R (array) [cpu]': ['saxpy_array.R', 'Implementation with array in [R](https://www.r-project.org/), a free software environment for statistical computing and graphics.'],
    'R (matrix) [cpu]': ['saxpy_matrix.R', 'Implementation with matrix in [R](https://www.r-project.org/), a free software environment for statistical computing and graphics.'],
    'R (data.frame) [cpu]': ['saxpy_dataframe.R', 'Implementation with `data.frame` in [R](https://www.r-project.org/), a free software environment for statistical computing and graphics.'],
    'R (data.table) [cpu]': ['saxpy_datatable.R', 'Implementation with `data.table` in [R](https://www.r-project.org/), a free software environment for statistical computing and graphics.'],

    'C++ loop [cpu]': ['saxpy_cpu.cpp', 'Plain C++ `for` loop'],
    'C++ CUDA [gpu]': ['saxpy_cuda.cpp', 'Low level implementation with the base NVidia [CUDA](https://developer.nvidia.com/cuda-toolkit) toolkit.'],
    'C++ Thrust [gpu]': ['saxpy_trust.cpp', 'A GPU implementation with NVidia [Thrust](https://thrust.github.io/), a parallel algorithms library which resembles the C++ Standard Template Library (STL). Thrust is included with [CUDA](https://developer.nvidia.com/cuda-toolkit) toolkit.'],
    'C++ cuBLAS [gpu]': ['saxpy_cublas.cpp', 'A GPU implementation with NVidia [cuBLAS](https://developer.nvidia.com/cublas), a fast GPU-accelerated implementation of the standard basic linear algebra subroutines (BLAS).'],
    'C++ Bulk [gpu]': ['saxpy_bulk.cpp', 'A GPU implementation with [Bulk](https://github.com/jaredhoberock/bulk), yet another parallel algorithms on top of CUDA.'],
    'C++ OCL [cpu]': ['saxpy_ocl1.cpp', 'Parallel programming with [OpenCL](https://en.wikipedia.org/wiki/OpenCL), a framework for writing programs that execute across heterogeneous platforms consisting of central processing units (CPUs), graphics processing units (GPUs), digital signal processors (DSPs), field-programmable gate arrays (FPGAs) and other processors or hardware accelerators.'],
    'C++ OCL [gpu]': ['saxpy_ocl1.cpp', 'Parallel programming with [OpenCL](https://en.wikipedia.org/wiki/OpenCL), a framework for writing programs that execute across heterogeneous platforms consisting of central processing units (CPUs), graphics processing units (GPUs), digital signal processors (DSPs), field-programmable gate arrays (FPGAs) and other processors or hardware accelerators.'],
    'C++ OMP [cpu]': ['saxpy_omp.cpp', 'Parallel programming with [OpenMP](http://www.openmp.org/). Only CPU version is implemented.'],
    'C++ TensorFlow [gpu]': ['saxpy_tf.cc', 'Implementation in C++ for GPU with [TensorFlow](https://www.tensorflow.org/), a deep learning library.'],

    'PyOCL [cpu]': ['saxpy_pyocl.py', 'CPU and GPU implementation with [PyOpenCL](https://mathema.tician.de/software/pyopencl/), the Python wrapper for [OpenCL](https://en.wikipedia.org/wiki/OpenCL).'],
    'PyOCL [gpu]': ['saxpy_pyocl.py', 'CPU and GPU implementation with [PyOpenCL](https://mathema.tician.de/software/pyopencl/), the Python wrapper for [OpenCL](https://en.wikipedia.org/wiki/OpenCL).'],
    'PyCUDA [gpu]': ['saxpy_pycuda.py', 'Implementation with [PyCUDA](https://mathema.tician.de/software/pycuda/), the Python wrapper for [CUDA](https://developer.nvidia.com/cuda-toolkit).'],
    'Py CNTK [gpu]': ['saxpy_cntk.py', 'Implementation for CPU and GPU with [CNTK](https://cntk.ai/), a deep learning library.'],
    'Py CNTK [cpu]': ['saxpy_cntk.py', 'Implementation for CPU and GPU with [CNTK](https://cntk.ai/), a deep learning library.'],
    'Py MXNet [cpu]': ['saxpy_mxnet.py', 'Implementation for CPU and GPU with [MXNet](https://mxnet.incubator.apache.org/), a deep learning library.'],
    'Py MXNet [gpu]': ['saxpy_mxnet.py', 'Implementation for CPU and GPU with [MXNet](https://mxnet.incubator.apache.org/), a deep learning library.'],
    'Py TensorFlow [cpu]': ['saxpy_tf.py', 'Implementation for CPU and GPU with [TensorFlow](https://www.tensorflow.org/), a deep learning library.'],
    'Py TensorFlow [gpu]': ['saxpy_tf.py', 'Implementation for CPU and GPU with [TensorFlow](https://www.tensorflow.org/), a deep learning library.'],

}


def create_chart0(spec, lang, output_dir):
    print('Processing {} "{}"'.format(spec['output'], spec['title']))

    plt.style.use('ggplot')

    png_file = spec['output']
    tmp_cols = spec.get('exclude', [])
    columns = spec.get('columns', [])
    drop_columns = []
    for col in tmp_cols:
        if "*" in col:
            col = col.replace("*", "")
            drop_columns.extend([c for c in known_columns.keys() if col in c])
        else:
            drop_columns.append(col)

    data = {}
    for serie in spec['series']:
        print("  Reading " + serie['data'] + "...")
        df = pd.read_csv(serie['data'], sep=',')
        if columns:
            df = df.loc[:, columns]
        if drop_columns:
            df = df.drop(drop_columns, axis=1, errors='ignore')
        df = df.mean()
        data[serie['title']] = df

    df = pd.DataFrame(data, index=data[next(iter(data))].index)

    # df will look something like this:
    """
                             Linux       Windows
    Python loop [cpu]  11934.702444  19970.356846
    Py Numpy [cpu]        77.160597    124.999094
    """

    df.sort_values(df.columns[0], ascending=False, inplace=True)
    # print(df)
    pivot_col = df.iloc[-1].idxmin()
    # print('  Lowest column is {}'.format(pivot_col))
    # print('  Lowest value is {}'.format(df.iloc[-1, :][pivot_col]))
    rel = df / df.iloc[-1, :][pivot_col]
    max_value = rel.max(axis=1).iloc[0]
    # print("max_value:", max_value)

    BAR_HEIGHT = 0.15
    BAR_SPACING = 0.05
    SERIES_CNT = len(df.columns)
    TOTAL_HEIGHT = BAR_HEIGHT * SERIES_CNT + BAR_SPACING

    y_pos = np.arange(len(df.index)) * TOTAL_HEIGHT
    # print("y_pos:", y_pos)

    fig = plt.figure(figsize=(7, 2 + TOTAL_HEIGHT * len(df.index)))
    ax = plt.subplot(111)
    colors = ['#009900', '#ff8000', '#999900', '#00999999']
    for i, col in enumerate(df.columns):
        x = y_pos + (i * BAR_HEIGHT)
        y = rel[col]
        color = spec['series'][i].get('color', colors[i])
        ax.barh(x, y, height=BAR_HEIGHT, align='center',
                color=color, ecolor='black', label=col)
        for j, row in enumerate(df.index):
            txt = '%.1f ms    (%.1fx)' % (df.loc[df.index[j], col], y[j])
            w = len(txt) * max_value * 0.012
            xpos = y[j] - w
            shift = 0
            yshift = 0.02
            if xpos > 0:
                ax.text(xpos, x[j] + yshift, txt,
                        fontdict={'size': 8}, color='w')
            else:
                ax.text(y[j] + max_value * 0.01, x[j] + yshift,
                        txt, fontdict={'size': 8}, color=colors[i])

    ax.text(rel.iloc[0, 0] * 4 / 8, y_pos[-1] + BAR_HEIGHT / 2 - 0.01,
            '©%d SAXPY Benchmark' % datetime.date.today().year,
            fontdict={'size': 8}, color='grey')
    ax.text(rel.iloc[0, 0] * 4 / 8, y_pos[-1] + BAR_HEIGHT / 2 - 0.01 + 0.01 * (len(df.index) * SERIES_CNT),
            'https://github.com/bennylp/saxpy-benchmark',
            fontdict={'size': 8}, color='grey')

    yticks = y_pos + (SERIES_CNT - 1) * BAR_HEIGHT / 2
    ax.set_yticks(yticks)
    # print("yticks:", yticks)

    ax.set_yticklabels(df.index, fontdict={'size': 9}, color="#202020")
    ax.invert_yaxis()  # labels read top-to-bottom
    label = 'Waktu (relatif thd. tercepat)' if lang == 'id' else 'Time (relative to fastest)'
    ax.set_xlabel(label, fontdict={'size': 11}, color="#202020")
    ax.set_title(spec.get('title-' + lang)
                 or spec['title'], fontdict={'size': 13})
    ax.legend(loc=spec.get('legend') or 'right',
              title='Legenda:' if lang == 'id' else 'Legend:')

    ax.grid(color='w')
    plt.savefig(output_dir + '/' + png_file, bbox_inches='tight')
    return


def create_chart(spec, output_md, use_rel=True):
    print('Processing {} "{}"'.format(spec['data'], spec['title']))

    plt.style.use('ggplot')

    png_file = os.path.splitext(spec['data'])[0] + ".png"

    output_md += "\n\n# {}\n\n".format(spec['title'])
    output_md += "Submitted by {} on {}\n\n".format(spec['by'], spec['date'])

    if spec['remarks']:
        output_md += spec['remarks'] + "\n\n"

    details = spec.get('details')
    if details:
        output_md += "\n### Specs\n\n"
        output_md += "|    |    |\n"
        output_md += "|----|----|\n"
        for row in details:
            output_md += "| {} | {} |\n".format(row[0], row[1])
        output_md += "\n"

    df = pd.read_csv(spec['data'])
    m = df.mean().sort_values(ascending=False)
    for idx in m.index:
        if idx not in known_columns:
            sys.stderr.write("Error: unrecognized column name '{}'. Standard names are required, choose one of:\n - {}\n".format(
                idx, '\n - '.join(sorted(known_columns.keys()))))
            sys.exit(1)
    rel = m / m[-1]
    std = df.std()

    output_md += "\n ### Result Details\n\n"
    output_md += "| Test   | Mean Time (ms) | StdDev (ms) | Time (rel)\n"
    output_md += "|--------| --------: | --------: | --------: |\n"

    for idx in m.index:
        output_md += "| {} | {} | {} | {} |\n".format(idx, "%.3f" % m[idx], "%.3f" % std[idx],
                                                      "%.2fx" % rel[idx])

    output_md += "\n"

    output_md += "\n### Result\n\n"
    output_md += '![{}]({}?raw=true "{}")\n\n'.format(png_file,
                                                      png_file, png_file)

    # Remove outliers as we don't support broken/discontinuous axis yet
    notes = ""
    for i in range(len(m)):
        if m[i] / m[-1] > 20:
            # print('** Warning: removing outlier "%s" from the chart' % m.index[i])
            notes += '- **outlier "{}" is removed from the chart because it is {}x slower**\n'.format(
                m.index[i], int(m[i] / m[-1]))
        else:
            break

    if notes:
        output_md += "\n### Notes\n\n" + notes

    m = m[i:]
    rel = rel[i:]

    BAR_HEIGHT = 0.3

    tests = m.index
    y_pos = np.arange(len(tests)) * BAR_HEIGHT
    y = rel
    xlims = None

    fig = plt.figure(figsize=(9, 2 + BAR_HEIGHT * len(m)))
    if xlims:
        ax = brokenaxes(xlims=xlims, hspace=.05)
    else:
        ax = plt.subplot(111)

    ax.barh(y_pos, y, height=BAR_HEIGHT - 0.15,
            align='center', color='green', ecolor='black')

    for i, v in enumerate(y):
        # ax.text(v, i, '%.3fx' % v)
        ax.text(max(0, v - 0.4), y_pos[i] + 0.03,
                '%.1fx' % v, fontdict={'size': 8}, color='w')
        ax.text(v + 0.1, y_pos[i] + 0.03, '%.1f ms' %
                m[i], fontdict={'size': 6}, color='grey')

    ax.text(y[0] * 5 / 8, y_pos[-1], '©%d SAXPY Benchmark' % datetime.date.today().year,
            fontdict={'size': 8}, color='grey')
    ax.text(y[0] * 5 / 8, y_pos[-1] + 0.1, 'https://github.com/bennylp/saxpy-benchmark',
            fontdict={'size': 8}, color='grey')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tests, fontdict={'size': 9}, color="#202020")
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Time (relative to fastest test)',
                  fontdict={'size': 11}, color="#202020")
    ax.set_title(spec['title'], fontdict={'size': 13})

    ax.grid(color='w')
    plt.savefig(png_file, bbox_inches='tight')

    output_md += "\n\n"
    return output_md


def create_front_page():
    def create_anchor(title):
        a = ""
        last = "-"
        for c in title:
            c = c.lower()
            if c.isalnum():
                a += c
                last = c
            elif c in ["&"]:
                a += "-"
                last = "-"
            elif c in ["."]:
                pass
            elif last != "-":
                a += "-"
                last = "-"
        if a[-1] == "-":
            a = a[:-1]
        return a

    chart_specs = json.loads(open("charts.json").read())
    result_specs = json.loads(open("result_specs.json").read())

    doc = ""
    doc += "# SAXPY CPU and GPGPU Benchmarks\n\n"
    doc += "**Table of Contents**:\n\n"
    doc += "- [Benchmarks](#benchmarks)\n"
    doc += "- [Results](#results)\n"
    for spec in chart_specs:
        doc += "   - [{}](#{})\n".format(spec['title'],
                                         create_anchor(spec['title']))
    doc += "- [Machine Specifications](#machine-specifications)\n"
    for spec in result_specs:
        doc += "   - [{}](#{})\n".format(spec['title'],
                                         create_anchor(spec['title']))
    doc += "\n\n"

    doc += "# Benchmarks\n\n"
    doc += "The following benchmarks are implemented:\n\n"
    for col in known_columns.keys():
        doc += "- **{}** ([src/{}](src/{}))\n".format(col,
                                                      known_columns[col][0], known_columns[col][0])
        doc += "  {}\n\n".format(known_columns[col][1])
    doc += "\n"

    doc += "# Results\n\n"
    for spec in chart_specs:
        doc += "## " + spec['title'] + "\n\n"
        for remark in spec.get('remarks', []):
            doc += remark + "\n\n"
        if 'exclude' in spec:
            doc += "**Excluded** from this chart:\n"
            for col in spec.get('exclude', []):
                if col not in known_columns:  # no need to print wildcards (e.g. *cpu*)
                    continue
                doc += "- {} ([src/{}](src/{}))\n".format(col,
                                                          known_columns[col][0], known_columns[col][0])
            doc += "\n"
        if 'columns' in spec:
            for col in sorted(spec.get('columns', [])):
                doc += "- {} ([src/{}](src/{}))\n".format(col,
                                                          known_columns[col][0], known_columns[col][0])
            doc += "\n"

        png_file = "results/charts-en/" + spec['output']
        doc += '![{}]({}?raw=true "{}")\n\n'.format(png_file,
                                                    png_file, png_file)

    doc += "\n\n"
    doc += "# Machine Specifications\n"
    for spec in result_specs:
        doc += "## " + spec['title'] + "\n\n"
        if spec.get('remarks'):
            doc += spec['remarks'] + '\n\n'
        doc += "|    |    |\n"
        doc += "|----|----|\n"
        for row in spec['details']:
            doc += "| {} | {} |\n".format(row[0], row[1])
        doc += "\n"
    doc += "\n"

    fname = "../front-page.md"
    print("Creating " + fname)
    out = open(fname, "wt")
    out.write(doc)
    out.close()


README_MD = """
# Results

This page is autogenerated from the results recorded in CSVs and [result_specs.json](result_specs.json)
file. Feel free to submit a result:
1. Record your test results in a CSV file. Repeat each test at least 5 times. See other csv files for samples.
2. Create an entry in [result_specs.json](result_specs.json)
3. Create pull request

When creating a CSV, please standardize the column names to the following, otherwise
the CSV will be rejected by `create_charts.py`. The names are standardized in case we want to do
some statistics with them in the future:
"""

if __name__ == "__main__":
    spec_file = None
    lang = 'en'
    cmd = None

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--lang':
            i += 1
            lang = sys.argv[i]
        elif sys.argv[i] in ['cmp', 'report']:
            cmd = sys.argv[i]
        else:
            spec_file = sys.argv[i]
        i += 1

    if cmd == 'cmp':
        specs = json.loads(open(spec_file).read())
        for spec in specs:
            create_chart0(spec, lang, 'charts-' + lang)
    elif cmd == "report":
        create_front_page()
    if not cmd:
        specs = json.loads(open('result_specs.json').read())

        output_md = README_MD
        output_md += "".join(["- %s\n" %
                              nm for nm in sorted(known_columns.keys())])
        output_md += "\n"

        for spec in specs:
            output_md = create_chart(spec, output_md)

        print('Writing README.md..')
        f = open('README.md', 'w')
        f.write(output_md)
        f.close()
