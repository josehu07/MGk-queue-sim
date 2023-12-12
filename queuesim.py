import argparse
import math
import statistics
import pickle
from multiprocess import Pool
import simpy
import numpy as np
from scipy import stats
from scipy.special import gammaln, logsumexp
from scipy.optimize import root_scalar, fsolve
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Distribution(object):
    def __init__(self):
        self.disp = "Base"
        self.mean = -1
        self.stdev = -1

    def name(self):
        return "base"

    def sample(self, rng):
        raise RuntimeError("calling .sample() on base class")


class DistExponential(Distribution):
    def __init__(self, rate):
        super().__init__()
        self.disp = "Exponential"
        self.scale = 1.0 / rate
        self.mean = self.scale
        self.stdev = self.scale
        self.cv2 = 1.0

    def name(self):
        return "exponential"

    def sample(self, rng):
        return rng.exponential(scale=self.scale)


class DistLognormal(Distribution):
    def __init__(self, target_m, target_cv2):
        super().__init__()
        self.disp = "Lognormal"
        self.mean = target_m
        self.stdev = target_m * math.sqrt(target_cv2)
        self.cv2 = target_cv2
        self.normal_si = math.sqrt(math.log(target_cv2 + 1))
        self.normal_mu = math.log(target_m) - (self.normal_si**2 / 2.0)
        # print(
        #     self.name(),
        #     (self.mean, self.stdev**2),
        #     stats.lognorm(self.normal_si, scale=math.exp(self.normal_mu)).stats(),
        # )

    def name(self):
        return f"lognormal-{self.mean:.1f}-{self.cv2:.1f}"

    def sample(self, rng):
        return rng.lognormal(mean=self.normal_mu, sigma=self.normal_si)


class DistTruncNormal(Distribution):
    @staticmethod
    def conversion(mean, stdev):
        """
        Numerically look for parameters given mean and stdev.
        Ref: https://stats.stackexchange.com/questions/408171
        """
        p_phi = lambda z: (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z**2)
        c_phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))

        def eqs(p):
            m, s = p
            r = m / s
            eq1 = m + s * (p_phi(r) / (1 - c_phi(-r))) - mean
            eq2 = (
                s**2
                * (
                    1
                    - ((m * p_phi(r) / s) / (1 - c_phi(-r)))
                    - (p_phi(r) / (1 - c_phi(-r))) ** 2
                )
                - stdev**2
            )
            return (eq1, eq2)

        x, _, ier, mesg = fsolve(
            eqs, (mean, stdev), maxfev=2000, xtol=1e-16, full_output=True
        )
        # if ier != 1:
        #     print(mesg)
        m, s = x

        return m, s

    def __init__(self, target_m, target_cv2):
        super().__init__()
        self.disp = "TruncNormal"
        self.mean = target_m
        self.stdev = target_m * math.sqrt(target_cv2)
        self.cv2 = target_cv2
        self.normal_mu, self.normal_si = DistTruncNormal.conversion(
            self.mean, self.stdev
        )
        r = self.normal_mu / self.normal_si
        self.dist = stats.truncnorm(
            -r, np.inf, loc=self.normal_mu, scale=self.normal_si
        )
        # print(
        #     self.name(),
        #     (self.mean, self.stdev**2),
        #     (self.normal_mu, self.normal_si),
        #     self.dist.stats(),
        # )

    def name(self):
        return f"truncnormal-{self.mean:.1f}-{self.cv2:.1f}"

    def sample(self, rng):
        if self.normal_mu < -5.0:  # while-loop might be prohibitively slow
            return self.dist.rvs(random_state=rng)
        else:
            while True:
                t = rng.normal(loc=self.normal_mu, scale=self.normal_si)
                if t >= 0:
                    return t


class DistWeibull(Distribution):
    @staticmethod
    def conversion(mean, stdev):
        """
        Numerically look for parameters given mean and stdev.
        Ref: https://github.com/scipy/scipy/issues/12134
        """
        log_mean, log_std = np.log(mean), np.log(stdev)

        def r(c):
            logratio = (  # np.pi*1j is the log of -1
                logsumexp([gammaln(1 + 2 / c) - 2 * gammaln(1 + 1 / c), np.pi * 1j])
                - 2 * log_std
                + 2 * log_mean
            )
            return np.real(logratio)

        res = root_scalar(
            r, method="bisect", bracket=[1e-300, 1e300], maxiter=2000, xtol=1e-16
        )
        assert res.converged
        c = res.root
        scale = np.exp(log_mean - gammaln(1 + 1 / c))
        return c, scale

    def __init__(self, target_m, target_cv2):
        super().__init__()
        self.disp = "Weibull"
        self.mean = target_m
        self.stdev = target_m * math.sqrt(target_cv2)
        self.cv2 = target_cv2
        self.alpha, self.scale = DistWeibull.conversion(self.mean, self.stdev)
        # print(
        #     self.name(),
        #     (self.mean, self.stdev**2),
        #     stats.weibull_min(self.alpha, scale=self.scale).stats(),
        # )

    def name(self):
        return f"weibull-{self.mean:.1f}-{self.cv2:.1f}"

    def sample(self, rng):
        return self.scale * rng.weibull(self.alpha)


ARR_RATE = 9.0
ARR_DIST = DistExponential(ARR_RATE)

CV2_SRANGE = [0.1 * i for i in range(1, 11)]
CV2_LRANGE = [9 + 10 * i for i in range(10)]
MEAN_RANGE = [0.1 * i for i in range(1, 11)]
K_RANGE = [i for i in range(10, 16)]
G_K_LIST = (
    [(DistExponential(1.0), 10)]
    + [(DistLognormal(1.0, cv2), 10) for cv2 in CV2_SRANGE]
    + [(DistLognormal(1.0, cv2), 10) for cv2 in CV2_LRANGE]
    + [(DistLognormal(mean, 1.0), 10) for mean in MEAN_RANGE]
    + [(DistTruncNormal(1.0, cv2), 10) for cv2 in CV2_SRANGE]
    + [(DistTruncNormal(mean, 1.0), 10) for mean in MEAN_RANGE]
    + [(DistWeibull(1.0, cv2), 10) for cv2 in CV2_SRANGE]
    + [(DistWeibull(1.0, cv2), 10) for cv2 in CV2_LRANGE]
    + [(DistWeibull(mean, 1.0), 10) for mean in MEAN_RANGE]
    + [(DistExponential(1.0), k) for k in K_RANGE]
    + [(DistLognormal(1.0, 1.0), k) for k in K_RANGE]
    + [(DistTruncNormal(1.0, 1.0), k) for k in K_RANGE]
    + [(DistWeibull(1.0, 1.0), k) for k in K_RANGE]
)

NUM_ARRIVALS = 500000
NUM_TRIALS = 100


class MGkQueue(object):
    def __init__(self, rng, env, a, g, k, logging=False):
        self.rng = rng
        self.env = env
        self.system = simpy.Resource(env, capacity=k)
        self.cus_map = dict()
        self.next_id = 0
        self.a = a
        self.g = g
        self.results = []
        self.logging = logging

    def print_stats(self, extra=""):
        print(
            f" [{len(self.system.queue)}] "
            + f"{self.system.count:>2d}/{self.system.capacity:<2d} {extra}"
        )

    def customer(self, cus_id, req):
        # wait in queue if no counters available
        yield req
        start = self.cus_map[cus_id]
        del self.cus_map[cus_id]
        elapsed = self.env.now - start
        if self.logging:
            self.print_stats(extra=f"waited {elapsed:.3f}")
        self.results.append(elapsed)
        # being serviced at a counter
        yield self.env.timeout(self.g.sample(self.rng))
        self.system.release(req)

    def arrival(self):
        while True:
            if self.logging:
                self.print_stats(extra="arrive")
            # put a new customer in the form of a request on the resource
            self.cus_map[self.next_id] = self.env.now
            req = self.system.request()
            self.env.process(self.customer(self.next_id, req))
            # Poisson process: randomly wait for an exponential interval
            self.next_id += 1
            if self.next_id == NUM_ARRIVALS:
                break
            yield self.env.timeout(self.a.sample(self.rng))


def simulate(a, g, k, logging):
    rng = np.random.default_rng()
    env = simpy.Environment()
    queue = MGkQueue(rng, env, a, g, k, logging)
    env.process(queue.arrival())
    env.run()
    return sum(queue.results) / len(queue.results)


def do_simulations(logging):
    for g, k in G_K_LIST:
        print(f"g={g.name()} k={k}")
        with Pool() as pool:
            ws = pool.map(
                lambda _: simulate(ARR_DIST, g, k, logging),
                range(NUM_TRIALS),
            )
        with open(f"{g.name()}-{k}.pkl", "wb") as fpkl:
            pickle.dump(ws, fpkl)


def read_results():
    results = dict()
    for g, k in G_K_LIST:
        results[(g, k)] = dict()
        with open(f"{g.name()}-{k}.pkl", "rb") as fpkl:
            l = pickle.load(fpkl)
            # l.sort()
            # throw = int(len(l) * 0.05)  # remove 10% outliers
            # l = l[throw:-throw]
            avg = sum(l) / len(l)
            cil = 1.96 * statistics.stdev(l) / math.sqrt(len(l))  # 95% CI
            results[(g, k)] = (avg, cil)
    return results


def approximation(g, k):
    dep_rate = 1.0 / g.mean
    util = ARR_RATE / (k * dep_rate)
    denom_sum = sum(((k * util) ** i / math.factorial(i) for i in range(k)))
    prob = 1.0 / (1 + (1 - util) * (math.factorial(k) / ((k * util) ** k)) * denom_sum)
    w_MMk = prob / (k * dep_rate - ARR_RATE)
    cv = g.stdev / g.mean
    w_MGk = ((cv**2 + 1) / 2.0) * w_MMk
    return w_MGk


DIST_COLOR_STYLE = {
    "Approx.": ("black", "--"),
    "Lognormal": ("steelblue", "-"),
    "TruncNormal": ("green", "-"),
    "Weibull": ("orchid", "-"),
    "Exponential": ("orange", ":"),
}


def plot_cv2(results):
    plt.rcParams["figure.figsize"] = (7, 3)

    # small cv2
    DISTS_ORDER = ["Lognormal", "TruncNormal", "Weibull"]
    xs = CV2_SRANGE
    ysd = {"Approx.": []}
    yerrsd = dict()
    relasd = dict()
    for cv2 in xs:
        for g, k in results:
            if g.mean == 1.0 and g.cv2 == cv2 and k == 10 and g.disp == DISTS_ORDER[0]:
                ysd["Approx."].append(approximation(g, k))
                break
    for dist in DISTS_ORDER:
        ys, yerrs, relas = [], [], []
        for cv2 in xs:
            if dist == "TruncNormal" and cv2 == 1.0:
                continue
            for g, k in results:
                if g.mean == 1.0 and g.cv2 == cv2 and k == 10 and g.disp == dist:
                    y, yerr = results[(g, k)]
                    relas.append((ysd["Approx."][len(ys)] - y) / y)
                    ys.append(y)
                    yerrs.append(yerr)
                    break
        ysd[dist] = ys
        yerrsd[dist] = yerrs
        relasd[dist] = relas

    ax = plt.subplot(221)
    for dist in DISTS_ORDER:
        color, style = DIST_COLOR_STYLE[dist]
        plt.plot(
            xs if dist != "TruncNormal" else xs[:-1],
            relasd[dist],
            color=color,
            linestyle=style,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tick_params(bottom=False, labelbottom=False)
    plt.ylabel("Error ratio")

    ax = plt.subplot(223)
    for dist in DISTS_ORDER:
        color, style = DIST_COLOR_STYLE[dist]
        plt.errorbar(
            xs if dist != "TruncNormal" else xs[:-1],
            ysd[dist],
            yerr=yerrsd[dist],
            color=color,
            capsize=2.0,
            linestyle=style,
            label=dist,
        )
    color, style = DIST_COLOR_STYLE["Approx."]
    plt.errorbar(xs, ysd["Approx."], color=color, linestyle=style, label="Approx.")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(xs, [f".{int(cv2*10):1d}" if cv2 < 1.0 else str(int(cv2)) for cv2 in xs])
    plt.xlabel("CV² of G (small)")
    plt.ylabel("Avg. waiting time")
    handles, labels = ax.get_legend_handles_labels()

    # large cv2
    DISTS_ORDER = ["Lognormal", "Weibull"]
    xs = CV2_LRANGE
    ysd = {"Approx.": []}
    yerrsd = dict()
    relasd = dict()
    for cv2 in xs:
        for g, k in results:
            if g.mean == 1.0 and g.cv2 == cv2 and k == 10 and g.disp == DISTS_ORDER[0]:
                ysd["Approx."].append(approximation(g, k))
                break
    for dist in DISTS_ORDER:
        ys, yerrs, relas = [], [], []
        for cv2 in xs:
            for g, k in results:
                if g.mean == 1.0 and g.cv2 == cv2 and k == 10 and g.disp == dist:
                    y, yerr = results[(g, k)]
                    relas.append((ysd["Approx."][len(ys)] - y) / y)
                    ys.append(y)
                    yerrs.append(yerr)
                    break
        ysd[dist] = ys
        yerrsd[dist] = yerrs
        relasd[dist] = relas

    ax = plt.subplot(222)
    for dist in DISTS_ORDER:
        color, style = DIST_COLOR_STYLE[dist]
        plt.plot(xs, relasd[dist], color=color, linestyle=style)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tick_params(bottom=False, labelbottom=False)

    ax = plt.subplot(224)
    for dist in DISTS_ORDER:
        color, style = DIST_COLOR_STYLE[dist]
        plt.errorbar(
            xs, ysd[dist], yerr=yerrsd[dist], color=color, capsize=2.0, linestyle=style
        )
    color, style = DIST_COLOR_STYLE["Approx."]
    plt.errorbar(xs, ysd["Approx."], color=color, linestyle=style)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(xs, [f"{cv2:d}" for cv2 in xs])
    plt.xlabel("CV² of G (large)")

    plt.figlegend(handles, labels, loc="center left", bbox_to_anchor=(0.79, 0.5))

    plt.gcf().align_labels()
    plt.subplots_adjust(left=0.12, bottom=0.15, top=0.95, right=0.78, wspace=0.22)
    plt.savefig("plot-cv2.pdf")
    plt.close()


def plot_m_k(results):
    plt.rcParams["figure.figsize"] = (7, 3)

    # mean
    DISTS_ORDER = ["Lognormal", "TruncNormal", "Weibull"]
    xs = MEAN_RANGE
    ysd = {"Approx.": []}
    yerrsd = dict()
    relasd = dict()
    for mean in xs:
        for g, k in results:
            if g.mean == mean and g.cv2 == 1.0 and k == 10 and g.disp == DISTS_ORDER[0]:
                ysd["Approx."].append(approximation(g, k))
                break
    for dist in DISTS_ORDER:
        ys, yerrs, relas = [], [], []
        for mean in xs:
            for g, k in results:
                if g.mean == mean and g.cv2 == 1.0 and k == 10 and g.disp == dist:
                    y, yerr = results[(g, k)]
                    relas.append((ysd["Approx."][len(ys)] - y) / y)
                    ys.append(y)
                    yerrs.append(yerr)
                    break
        ysd[dist] = ys
        yerrsd[dist] = yerrs
        relasd[dist] = relas

    ax = plt.subplot(221)
    for dist in DISTS_ORDER:
        color, style = DIST_COLOR_STYLE[dist]
        plt.plot(xs, relasd[dist], color=color, linestyle=style)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tick_params(bottom=False, labelbottom=False)
    plt.ylabel("Error ratio")

    ax = plt.subplot(223)
    for dist in DISTS_ORDER:
        color, style = DIST_COLOR_STYLE[dist]
        plt.errorbar(
            xs,
            ysd[dist],
            yerr=yerrsd[dist],
            color=color,
            capsize=2.0,
            linestyle=style,
        )
    color, style = DIST_COLOR_STYLE["Approx."]
    plt.errorbar(xs, ysd["Approx."], color=color, linestyle=style)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(xs, [f".{int(m*10):1d}" if m < 1.0 else str(int(m)) for m in xs])
    plt.xlabel("Mean of G")
    plt.ylabel("Avg. waiting time")

    # k
    DISTS_ORDER = ["Lognormal", "TruncNormal", "Weibull", "Exponential"]
    xs = K_RANGE
    ysd = {"Approx.": []}
    yerrsd = dict()
    relasd = dict()
    for xk in xs:
        for g, k in results:
            if g.mean == 1.0 and g.cv2 == 1.0 and k == xk and g.disp == DISTS_ORDER[0]:
                ysd["Approx."].append(approximation(g, k))
                break
    for dist in DISTS_ORDER:
        ys, yerrs, relas = [], [], []
        for xk in xs:
            for g, k in results:
                if g.mean == 1.0 and g.cv2 == 1.0 and k == xk and g.disp == dist:
                    y, yerr = results[(g, k)]
                    relas.append((ysd["Approx."][len(ys)] - y) / y)
                    ys.append(y)
                    yerrs.append(yerr)
                    break
        ysd[dist] = ys
        yerrsd[dist] = yerrs
        relasd[dist] = relas

    ax = plt.subplot(222)
    for dist in DISTS_ORDER:
        color, style = DIST_COLOR_STYLE[dist]
        plt.plot(xs, relasd[dist], color=color, linestyle=style)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tick_params(bottom=False, labelbottom=False)

    ax = plt.subplot(224)
    for dist in DISTS_ORDER:
        color, style = DIST_COLOR_STYLE[dist]
        plt.errorbar(
            xs,
            ysd[dist],
            yerr=yerrsd[dist],
            color=color,
            capsize=2.0,
            linestyle=style,
            label=dist,
        )
    color, style = DIST_COLOR_STYLE["Approx."]
    plt.errorbar(xs, ysd["Approx."], color=color, linestyle=style, label="Approx.")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(xs, [f"{k:d}" for k in xs])
    plt.xlabel("#Counters k")
    handles, labels = ax.get_legend_handles_labels()

    plt.figlegend(handles, labels, loc="center left", bbox_to_anchor=(0.79, 0.5))

    plt.gcf().align_labels()
    plt.subplots_adjust(left=0.11, bottom=0.15, top=0.95, right=0.78, wspace=0.25)
    plt.savefig("plot-m_k.pdf")
    plt.close()


def plot_results(results):
    for g, k in G_K_LIST:
        approx = approximation(g, k)
        avg_sim, cil_sim = results[(g, k)]
        print(
            f"g={g.name()} k={k}  approx: {approx:.3f}  "
            + f"sim: {avg_sim:.3f} ±{cil_sim:.3f}"
        )
    plot_cv2(results)
    plot_m_k(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l",
        "--logging",
        action="store_true",
        help="if set, print log during simulation",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="if set, do the plotting phase",
    )
    args = parser.parse_args()

    if not args.plot:
        do_simulations(args.logging)
    else:
        results = read_results()
        plot_results(results)
