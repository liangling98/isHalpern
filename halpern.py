import numpy as np
import matplotlib.pyplot as plt


class WDRO:
    def __init__(self, A, lam, lb, ub, train_data, theta) -> None:
        self.A = A
        self.At = np.transpose(A)
        self.lam = lam
        self.lb = lb
        self.ub = ub
        self.d = A.shape[0]
        self.n = A.shape[1]
        self.train_data = train_data
        self.train_data_vec = np.reshape(train_data, (-1))
        self.N = train_data.shape[0]  # each row represents a data point
        self.theta = theta
        self.Lip = self.computeLip()
        self.alpha = 2.0 / self.Lip

    def computeLip(self):
        Q = np.block(
            [
                [self.At @ self.A, -self.At],
                [-self.A, (1.0 - self.lam) * np.identity(self.d)],
            ]
        )

        return np.linalg.norm(Q, ord=2)

    def checkFeasibiltyX(self, x, tol=0.0):
        return (
            abs(np.sum(x) - 1) < tol
            and all(x >= self.lb - tol)
            and all(x <= self.ub + tol)
        )

    def projX(self, x, max_iter_proj=1000000, tol_proj=1e-8):
        # alternating projection
        iter = 0          
        while (not self.checkFeasibiltyX(x, tol_proj)) and iter < max_iter_proj:
            x = np.clip(x, self.lb, self.ub)
            x = x - (np.sum(x) - 1.0) / self.n
            iter = iter + 1

        return x, iter

    def projY(self, y):
        # analytical expression
        y_tmp = y - self.train_data_vec
        norm_y_tmp = np.linalg.norm(y_tmp)

        return (
            self.train_data_vec
            + ((self.N) ** 0.5 * self.theta * min(1.0, norm_y_tmp)) * y_tmp
        )

    def evalGrad(self, x, y, n_samples):
        Y = np.reshape(y, (self.N, self.d))
        Ax = self.A @ x

        if n_samples < self.N:
            # stochatic gradient estimator
            idx = np.random.choice(
                self.N, n_samples, replace=False
            )  # without replacement
            grad_Lx = self.At @ Ax - (Y[idx, :] @ self.A).sum(axis=0) / n_samples
            Y_tmp = np.zeros_like(Y)
            Y_tmp[idx, :] = ((1.0 - self.lam) / n_samples) * Y[idx, :] - Ax / n_samples
            grad_Ly = np.reshape(Y_tmp, (-1))
        else:
            # full gradient
            grad_Lx = self.At @ Ax - (Y @ self.A).sum(axis=0) / self.N
            grad_Ly = np.reshape(((1.0 - self.lam) / self.N) * Y - Ax / self.N, (-1))

        return grad_Lx, grad_Ly

    def evalPAGE(self, x, x_old, y, y_old, grad_Lx_last, grad_Ly_last, n_samples):
        Y = np.reshape(y, (self.N, self.d))
        Y_old = np.reshape(y_old, (self.N, self.d))
        Ax = self.A @ x
        Ax_old = self.A @ x_old

        idx = np.random.choice(self.N, n_samples, replace=False)  # without replacement
        grad_Lx = self.At @ Ax - (Y[idx, :] @ self.A).sum(axis=0) / n_samples
        grad_Lx_old = (
            self.At @ Ax_old - (Y_old[idx, :] @ self.A).sum(axis=0) / n_samples
        )
        Y_tmp = np.zeros_like(Y)
        Y_tmp_old = np.zeros_like(Y)
        Y_tmp[idx, :] = ((1.0 - self.lam) / n_samples) * Y[idx, :] - Ax / n_samples
        Y_tmp_old[idx, :] = ((1.0 - self.lam) / n_samples) * Y_old[
            idx, :
        ] - Ax_old / n_samples
        grad_Ly = np.reshape(Y_tmp, (-1))
        grad_Ly_old = np.reshape(Y_tmp_old, (-1))

        return (
            grad_Lx_last + grad_Lx - grad_Lx_old,
            grad_Ly_last + grad_Ly - grad_Ly_old,
        )

    def evalG(self, x, y, grad_Lx, grad_Ly, max_iter_proj=1000000, tol_proj=1e-8):
        x_input = x - self.alpha * grad_Lx
        y_input = y + self.alpha * grad_Ly

        x_proj, iter_proj = self.projX(
            x_input, max_iter_proj=max_iter_proj, tol_proj=tol_proj
        )
        y_proj = self.projY(y_input)

        Gx = (x - x_proj) / self.alpha
        Gy = (y - y_proj) / self.alpha

        return Gx, Gy, iter_proj

    def normG(self, x, y):
        grad_Lx, grad_Ly = self.evalGrad(x, y, n_samples=self.N)
        Gx, Gy, iter_proj = self.evalG(x, y, grad_Lx, grad_Ly, tol_proj=1e-12)
        err = (np.linalg.norm(Gx) ** 2 + np.linalg.norm(Gy) ** 2) ** 0.5

        return err

    def halpern(
        self,
        x=None,
        y=None,
        sample_ratio=1.0,
        max_grad=1000000,
        tol=1e-2,
        exact_proj=True,
        PAGE=False,
        PAGE_ratio_1=0.1,
        PAGE_ratio_2=0.5,
    ):
        # initial points
        if x is None:
            x = np.random.rand(self.n)
            x = x / np.sum(x)
        x, iter_proj = self.projX(x, tol_proj=1e-12)
        total_proj = []
        total_samples = []
        norm_G = []
        total_proj.append(iter_proj)

        if y is None:
            y = self.train_data_vec

        x0 = x
        y0 = y

        norm_G.append(self.normG(x, y))
        total_samples.append(0)

        # main loop
        num_grad = 0
        iter = 0
        while num_grad < max_grad:
            if PAGE and iter > 0:
                x_old = x
                y_old = y
                grad_Lx_last = grad_Lx
                grad_Ly_last = grad_Ly

            # construct gradient estimator
            if PAGE:
                if iter % 10 == 0:
                    n_samples = int(self.N * PAGE_ratio_2)
                    grad_Lx, grad_Ly = self.evalGrad(x, y, n_samples=n_samples)
                    total_samples.append(n_samples)
                    num_grad = num_grad + n_samples
                else:
                    n_samples = int(self.N * PAGE_ratio_1)
                    grad_Lx, grad_Ly = self.evalPAGE(
                        x, x_old, y, y_old, grad_Lx_last, grad_Ly_last, n_samples
                    )
                    total_samples.append(n_samples * 2)
                    num_grad = num_grad + n_samples * 2
            else:
                n_samples = int(self.N * sample_ratio)
                grad_Lx, grad_Ly = self.evalGrad(x, y, n_samples=n_samples)
                total_samples.append(n_samples)
                num_grad = num_grad + n_samples

            if exact_proj:
                tol_proj = 1e-12
            else:
                tol_proj = 0.5 * tol / (iter + 1) ** 0.5

            Gx, Gy, iter_proj = self.evalG(x, y, grad_Lx, grad_Ly, tol_proj=tol_proj)

            beta_k = 1.0 / (iter + 2)
            eta_k = (1.0 - beta_k) / self.Lip
            x = beta_k * x0 + (1.0 - beta_k) * x - eta_k * Gx
            y = beta_k * y0 + (1.0 - beta_k) * y - eta_k * Gy

            norm_G.append(self.normG(x, y))
            total_proj.append(iter_proj)
            iter = iter + 1

        return norm_G, total_samples, total_proj, iter

    # projected gradient descent-ascent algorithm
    def pgda(self, x=None, y=None, sample_ratio=1.0, max_grad=1000000):
        # initial points
        if x is None:
            x = np.random.rand(self.n)
            x = x / np.sum(x)
        x, iter_proj = self.projX(x, tol_proj=1e-12)

        if y is None:
            y = self.train_data_vec

        n_samples = int(self.N * sample_ratio)
        total_proj = []
        total_samples = []
        norm_G = []
        total_proj.append(iter_proj)
        norm_G.append(self.normG(x, y))
        total_samples.append(0)

        # main loop
        num_grad = 0
        iter = 0
        step_size = 1.75 / self.Lip
        while num_grad < max_grad:
            grad_Lx, grad_Ly = self.evalGrad(x, y, n_samples=n_samples)
            total_samples.append(n_samples)

            x_input = x - step_size * grad_Lx
            y_input = y + step_size * grad_Ly

            x, iter_proj = self.projX(x_input, tol_proj=1e-12)
            y = self.projY(y_input)

            total_proj.append(iter_proj)
            err = self.normG(x, y)
            norm_G.append(err)
            num_grad = num_grad + n_samples
            iter = iter + 1
            # print(iter, err)

        return norm_G, total_samples, total_proj, iter


if __name__ == "__main__":
    # generate data
    np.random.seed(0)  # reproducibility
    dd = [50]
    for d in dd:
        for k in [1, 2, 5, 10]:
            n = d * k 
            N = 200 * d
            A = np.random.rand(d, n)
            # normalize A
            for i in range(n):
                A[:, i] = A[:, i] / np.linalg.norm(A[:, i])
            lam = 1.0
            
            # initial point
            x = np.random.rand(n)
            x = x / np.sum(x)
            noise_level = 2.5e-1
            lb = x - np.random.rand(n) * noise_level
            ub = x + np.random.rand(n) * noise_level
            train_data = np.random.rand(N, d)
            theta = 1e-2
            
            # construct model
            model = WDRO(A, lam, lb, ub, train_data, theta)

            max_grad = 50 * N
            tol = 1e-2
            
            

            # evaluate performance
            run_HI = True
            run_iHI = True
            run_isHI = True
            run_pgda = True

            # exact Halpern iteration
            if run_HI:
                norm_G_HI, total_samples_HI, total_proj_HI, iter_HI = model.halpern(
                    x=x,
                    sample_ratio=1.0,
                    max_grad=max_grad,
                    tol=tol,
                    exact_proj=True,
                    PAGE=False,
                )
                avg_proj_HI = int(np.sum(total_proj_HI) / iter_HI)

            # inexact deterministic Halpern iteration
            if run_iHI:
                norm_G_iHI, total_samples_iHI, total_proj_iHI, iter_iHI = model.halpern(
                    x=x,
                    sample_ratio=1.0,
                    max_grad=max_grad,
                    tol=tol,
                    exact_proj=False,
                    PAGE=False,
                )
                avg_proj_iHI = int(np.sum(total_proj_iHI) / iter_iHI)

            # ineaxct stochastic Halpern iteration
            if run_isHI:
                norm_G_isHI, total_samples_isHI, total_proj_isHI, iter_isHI = model.halpern(
                    x=x,
                    sample_ratio=0.1,
                    max_grad=max_grad,
                    tol=tol,
                    exact_proj=False,
                    PAGE=True,
                    PAGE_ratio_1=1/N**0.7,
                    PAGE_ratio_2=1/N**0.3,
                )
                avg_proj_isHI = int(np.sum(total_proj_isHI) / iter_isHI)

            # projected gradient descent ascend algorithm
            if run_pgda:
                norm_G_pgda, total_samples_pgda, total_proj_pgda, iter_pgda = model.pgda(
                    x=x, sample_ratio=1.0, max_grad=max_grad
                )
                avg_proj_pgda = int(np.sum(total_proj_pgda) / iter_pgda)


            # plot results
            plt.figure()
            if run_HI:
                plt.semilogy(
                    np.cumsum(np.array(total_samples_HI)),
                    np.array(norm_G_HI),
                    marker="+",
                    label="HI" + f", #proj/iter: {avg_proj_HI}",
                )

            if run_iHI:
                plt.semilogy(
                    np.cumsum(np.array(total_samples_iHI)),
                    np.array(norm_G_iHI),
                    marker=".",
                    label="iHI" + f", #proj/iter: {avg_proj_iHI}",
                )

            if run_isHI:
                plt.semilogy(
                    np.cumsum(np.array(total_samples_isHI)),
                    np.array(norm_G_isHI),
                    marker="*",
                    label="isHI" + f", #proj/iter: {avg_proj_isHI}",
                )

            if run_pgda:
                plt.semilogy(
                    np.cumsum(np.array(total_samples_pgda)),
                    np.array(norm_G_pgda),
                    marker="s",
                    label="pgda" + f", #proj/iter: {avg_proj_pgda}",
                )

            plt.xlabel("#grad")
            plt.ylabel("$||G(x,y)||$")
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
            plt.savefig(f"fig-{d}-{n}.png", format="png")
