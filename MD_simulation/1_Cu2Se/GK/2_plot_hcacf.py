import os
import feather
import pandas as pd
import xarray as xr
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from pathlib import Path


def get_random_numbers(temp, directory=None):
    p = Path(".") if not directory else Path(directory)
    random_numbers = [i.stem.split("-")[-1] for i in p.glob("*.stat") if f"-{temp:.1f}-" in i.name]
    return random_numbers


def get_filename(number, header, tail, temp="*", directory=None):
    p = Path(".") if not directory else Path(directory)
    print("to find: ", f"{header}-{temp:.1f}-{number}.{tail}")
    fn = list(p.glob(f"{header}-{temp:.1f}-{number}.{tail}"))[0].name
    return fn


def load_dat(numbers, temp):
    dat_hcacf = xr.Dataset()
    for ni in numbers:
        fn_hcacf = get_filename(ni, "hcacf", "feather", temp)
        print(fn_hcacf)
        # dat_hcacf[str(ni)] = pd.read_feather(fn_hcacf)
        dat_hcacf[str(ni)] = feather.read_dataframe(fn_hcacf)
    return dat_hcacf


def hcacf2kappa(hcacf, temp, V, delta=0.001, nevery=10):
    T = temp
    kB = 1.38e-23
    eV2J = 1.602176565e-19
    A2m = 1.0e-10
    ps2s = 1e-12
    convert = eV2J * eV2J / ps2s / A2m
    return integrate.cumtrapz(hcacf, axis=0, initial=0) * convert * V / kB / T ** 2 * delta * nevery


def cal_kappa(dat_hcacf, temp, delta=0.001, nevery=10):
    numbers = [i for i in list(dat_hcacf.data_vars.keys()) if "kappa" not in i and "mean" not in i]
    for k in numbers:
        hcacf_tmp = dat_hcacf[k]
        fn_stat = get_filename(k, "heatflux", "stat", temp)
        print("Read stat file from {}.\n".format(fn_stat))
        V = np.genfromtxt(fn_stat)[-1, 2]
        print(V)
        kappa_tmp = hcacf2kappa(hcacf_tmp, temp, V, delta, nevery)
        coords_tmp = hcacf_tmp.coords["dim_1"].values
        dat_hcacf[k + "_kappa"] = xr.DataArray(kappa_tmp, coords={"dim_1": coords_tmp})
    # cal mean hcacf, kappa
    dat_hcacf["mean_hcacf"] = sum(dat_hcacf[i] for i in numbers) / len(numbers)
    mean_kappa_1 = hcacf2kappa(dat_hcacf["mean_hcacf"], temp, V, delta, nevery)
    dat_hcacf["mean_kappa_1"] = xr.DataArray(mean_kappa_1, coords={"dim_1": coords_tmp})
    dat_hcacf["mean_kappa"] = sum(dat_hcacf[i + "_kappa"] for i in numbers) / len(numbers)
    return dat_hcacf


def plot_hcacf(dat_hcacf, temp, corre_t, delta=0.01, nevery=10):
    numbers = [i for i in list(dat_hcacf.data_vars.keys()) if "kappa" not in i and "mean" not in i]
    corre_N = int(corre_t / delta / nevery)
    time = np.array(dat_hcacf.indexes["dim_0"]) * delta * nevery
    fig, ax = plt.subplots()
    out_array=time[:corre_N]
    for i in numbers:
        ax.plot(
            time[:corre_N],
            dat_hcacf[i].sel(dim_1="JJ").values[:corre_N] / dat_hcacf[i].sel(dim_1="JJ").values[0],
            #dat_hcacf[i].sel(dim_1="JJ").values[:corre_N],
            "gray",
            linewidth=0.04,
        )
        out_array=np.vstack((out_array,dat_hcacf[i].sel(dim_1="JJ").values[:corre_N] / dat_hcacf[i].sel(dim_1="JJ").values[0]))
    ax.plot(
        time[:corre_N],
        dat_hcacf["mean_hcacf"].sel(dim_1="JJ").values[:corre_N] / dat_hcacf["mean_hcacf"].sel(dim_1="JJ").values[0],
        #dat_hcacf["mean_hcacf"].sel(dim_1="JJ").values[:corre_N],
        "r",
        label="T={}K".format(temp),
        lw = 1.5
    )
    out_array=np.vstack((out_array,dat_hcacf["mean_hcacf"].sel(dim_1="JJ").values[:corre_N] / dat_hcacf["mean_hcacf"].sel(dim_1="JJ").values[0]))
    np.savetxt("hcacf.dat",out_array.T,fmt="%6.4f")
    ax.set_ylabel("Normalized HCACF ")
    ax.set_xlabel("t (ps)")
    ll = ax.legend(loc=1)
    for text in ll.get_texts():
        text.set_color("red")
    fout = f"hcacf_{temp:.1f}.pdf"
    print(f"saving to {fout}")
    plt.savefig(fout)


def plot_rtc(dat_hcacf, temp, corre_t, delta=0.001, nevery=10):
    numbers = [i for i in list(dat_hcacf.data_vars.keys()) if "kappa" not in i and "mean" not in i]

    corre_N = int(corre_t / delta / nevery)
    time = np.array(dat_hcacf.indexes["dim_0"]) * delta * nevery
    fig, ax = plt.subplots()
    out_array2=time[:corre_N]
    for i in numbers:
        ax.plot(time[:corre_N], dat_hcacf[i + "_kappa"].sel(dim_1="JJ").values[:corre_N], "gray", linewidth=0.04)
        out_array2=np.vstack((out_array2,dat_hcacf[i + "_kappa"].sel(dim_1="JJ").values[:corre_N]))
    ax.plot(time[:corre_N:10], dat_hcacf["mean_kappa"].sel(dim_1="JJ").values[:corre_N:10], "r", label="T={}K".format(temp), lw=2.5)
    out_array2=np.vstack((out_array2,dat_hcacf["mean_kappa"].sel(dim_1="JJ").values[:corre_N]))
    np.savetxt("kappa.dat",out_array2.T,fmt="%6.4f")
    ax.set_ylabel("$\kappa$ (W/mK)")
    ax.set_xlabel("t (ps)")
    ax.set_ylim(-0,2.)
    ll = ax.legend(loc=1)
    for text in ll.get_texts():
        text.set_color("red")
    fout = f"kappa_{temp:.1f}.pdf"
    print(f"saving to {fout}")
    plt.savefig(fout)


def get_kappa(dat_hcacf_k, t1, t2, label="JJ", delta=0.001, nevery=10):
    numbers = [i for i in list(dat_hcacf_k.data_vars.keys()) if "kappa" not in i and "mean" not in i]
    N1 = int(t1 / delta / nevery)
    N2 = int(t2 / delta / nevery)
    kappa_converged = [np.mean(dat_hcacf_k[i + "_kappa"].sel(dim_1=label).values[N1:N2]) for i in numbers]
    kappa_average = np.mean(kappa_converged)
    kappa_error = np.std(kappa_converged) / np.sqrt(len(kappa_converged))
    return kappa_converged, kappa_average, kappa_error


def main(temp, t1, t2, delta=0.001, nevery=10, corre_t1=10, corre_t2=100, out_kappa="kappa_tmp.txt"):
    numbers = get_random_numbers(temp)
    dat_hcacf = load_dat(numbers, temp)
    dat_hcacf_k = cal_kappa(dat_hcacf, temp, nevery=nevery, delta=delta)
    plot_hcacf(dat_hcacf_k, temp, corre_t=corre_t1, nevery=nevery, delta=delta)
    plot_rtc(dat_hcacf_k, temp, corre_t=corre_t2, nevery=nevery, delta=delta)

    totalK = get_kappa(dat_hcacf_k, t1, t2, "JJ", nevery=nevery, delta=delta)
    totalK2 = get_kappa(dat_hcacf_k, t1, t2, "JJ2", nevery=nevery, delta=delta)
    totalK3 = get_kappa(dat_hcacf_k, t1, t2, "JJ3", nevery=nevery, delta=delta)
    # cuK = get_kappa(dat_hcacf_k, t1, t2, "J1J1", nevery=nevery, delta=delta)
    # seK = get_kappa(dat_hcacf_k, t1, t2, "J2J2", nevery=nevery, delta=delta)
    # cuseK = get_kappa(dat_hcacf_k, t1, t2, "J1J2", nevery=nevery, delta=delta)
    # secuK = get_kappa(dat_hcacf_k, t1, t2, "J2J1", nevery=nevery, delta=delta)

    print(totalK, totalK2, totalK3)
    print(f"kappa: [({totalK[1]},{totalK[2]}),({totalK2[1]},{totalK2[2]}),({totalK3[1]},{totalK3[2]}),({totalK[1]-totalK2[1]-totalK3[1]})]")


    # del dat_hcacf, dat_hcacf_k
    # # return dat_hcacf_k


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="calculate kappa")
    parser.add_argument("temp", type=int, help="temp")
    parser.add_argument("--delta", type=float, default=0.001, help="delta(ps)")
    parser.add_argument("--nevery", type=int, default=10, help="nevery")
    parser.add_argument("--t1", type=float, default=100, help="t1")
    parser.add_argument("--t2", type=float, default=500, help="t2")
    parser.add_argument("--corre_t1", type=float, default=200, help="corre_t1")
    parser.add_argument("--corre_t2", type=float, default=500, help="corre_t2")
    args = parser.parse_args()
    print(args)

    # main(400, t1=100, t2=500, corre_t1=200, corre_t2=500)
    main(
        args.temp,
        args.t1,
        args.t2,
        delta=args.delta,
        nevery=args.nevery,
        corre_t1=args.corre_t1,
        corre_t2=args.corre_t2,
    )
