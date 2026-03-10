import numpy as np
import scipy
import cvxpy as cp
import scipy.sparse

from sim_src.util import STATS_OBJECT, profile


class sdp_solver:
    def __init__(self, nit=100, rank_radio=2, alpha=1.):
        self.nit = nit
        self.rank_radio = rank_radio
        self.alpha = alpha # alpha is the scaling factor for the objective function, not used in this implementation

    def run_with_state(self, bs_iteration, Z, state):
        pass

    # [PRIO-ROUNDING] 新增 user_priority 入参：用于控制用户分配时隙前的处理顺序。
    # [BLE-TIMING] 新增 slot_mask 入参：用于限制用户可选时隙（如 BLE CI 对齐时隙）。
    def rounding(self,Z,gX,state,nattempt=10,user_priority=None,slot_mask=None):
        z_vec = None
        remainder = None
        for n in range(nattempt):
            z_vec, Z, remainder = self.rounding_one_attempt(Z,gX,state,user_priority=user_priority,slot_mask=slot_mask)
            if remainder == 0:
                return z_vec, Z, remainder
        return z_vec, Z, remainder

    # [PRIO-ROUNDING] 在每次 rounding 尝试中先分配高优先级用户，再分配低优先级用户。
    def rounding_one_attempt(self,Z,gX,state,user_priority=None,slot_mask=None):
        K = state[0].shape[0]
        D = gX.shape[1]
        S_gain = state[0].copy()
        Q_asso = state[1]
        h_max = state[2]
        S_gain.setdiag(0)
        S_gain.eliminate_zeros()
        S_gain_T_no_diag = S_gain.transpose()

        # A_sum = np.asarray(Q_asso.sum(axis=1)).ravel()
        # rank = np.argsort(-A_sum)

        S_gain_index = np.split(S_gain.indices, S_gain.indptr)[1:-1]
        Q_asso_index = np.split(Q_asso.indices, S_gain.indptr)[1:-1]

        not_assigned = np.ones(K, dtype=bool)

        # random_indices = np.random.choice(K, Z, replace=False)
        # randv = gX[random_indices]

        randv = np.random.randn(Z,D)
        randv = randv/np.linalg.norm(randv, axis=1, keepdims=True)

        norm_rank = np.linalg.norm(gX, axis=1)
        if slot_mask is not None:
            slot_mask = np.asarray(slot_mask, dtype=bool)
            if slot_mask.shape != (K, Z):
                raise ValueError("slot_mask shape must be (K, Z).")

        if user_priority is None:
            rank = np.argsort(-norm_rank)
        else:
            # [PRIO-ROUNDING] 排序规则：先按业务优先级从高到低，再按 ||gX_k|| 从大到小打破同优先级并列。
            prio = np.asarray(user_priority, dtype=float).ravel()
            if prio.shape[0] != K:
                raise ValueError("user_priority length must match number of users.")
            rank = np.lexsort((-norm_rank, -prio))
        # rank = np.argsort(np.random.randn(K))

        # randv = generate_rand_regular_simplex_with_Z_vertices(Z,D)

        inprod = np.matmul(randv,gX.transpose())
        sorted_indices = np.argsort(-inprod, axis=0)

        z_vec = np.zeros(K)

        gain_sum = []
        asso_sum = []
        slot_asn = []
        for z in range(Z):
            gain_sum.append(np.zeros(K))
            asso_sum.append(np.zeros(K))
            slot_asn.append([])

        user_sum = 0
        for kk in range(K):
            k = rank[kk]
            for zz in range(Z):
                z = sorted_indices[zz,k]
                if slot_mask is not None and not slot_mask[k, z]:
                    continue

                if not not_assigned[k]:
                    break

                # do interference check
                neighbor_index = np.intersect1d(np.array(slot_asn[z]),S_gain_index[k])
                neighbor_index = np.append(neighbor_index,k).astype(int)
                tmp_h = np.asarray(S_gain[k].toarray()).ravel()
                vio = (gain_sum[z][neighbor_index] + tmp_h[neighbor_index]) > h_max[neighbor_index]
                if np.any(vio == True):
                    continue

                # do association check
                neighbor_index = np.intersect1d(np.array(slot_asn[z]),Q_asso_index[k])
                neighbor_index = np.append(neighbor_index,k).astype(int)
                tmp_a = np.asarray(Q_asso[k].toarray()).ravel()
                vio = (asso_sum[z][neighbor_index] + tmp_a[neighbor_index]) >= 1
                if np.any(vio == True):
                    continue

                gain_sum[z] += np.asarray(S_gain[k].toarray()).ravel()
                asso_sum[z] += np.asarray(Q_asso[k].toarray()).ravel()
                slot_asn[z].append(k)

                user_sum += 1
                not_assigned[k] = False
                z_vec[k] = z
                break


        if not np.all(not_assigned == False):
            z_vec[not_assigned] = np.random.randint(Z,size = int(not_assigned.sum()))

        return z_vec, Z, np.sum(not_assigned)

class rand_sdp_solver(sdp_solver, STATS_OBJECT):
    def run_with_state(self, bs_iteration, Z, state):
        K = state[0].shape[0]
        randv = np.random.randn(K,Z*self.rank_radio)
        randv = randv/np.linalg.norm(randv, axis=1, keepdims=True)
        return True, randv
class admm_sdp_solver(sdp_solver, STATS_OBJECT):
    def __init__(self, nit=100, rank_radio=2, alpha=1.):
        sdp_solver.__init__(self, nit=nit, rank_radio=rank_radio, alpha=alpha)

    def run_with_state(self, bs_iteration, Z, state):
        ps_tic = self._get_tic()
        prob, X = self._setup_problem(Z,state[0],state[1],state[2])
        tim = self._get_tim(ps_tic)
        K = state[0].shape[0]
        self._add_np_log("admm_problem_setup",bs_iteration,np.array([Z,K,tim]))

        solving_tic = self._get_tic()
        prob.solve(solver=cp.SCS, max_iters=self.nit)
        if X.value is None:
            return True, np.random.randn(K,K)
        u, s, v = np.linalg.svd(X.value)
        rank = np.min([K , (Z-1)*self.rank_radio])
        X_half = u[:,0:rank] * np.sqrt(s[0:rank])
        tim = self._get_tim(solving_tic)
        self._add_np_log("admm_solve",bs_iteration,np.array([Z,K,tim]))
        return True, X_half

    def _setup_problem(self, Z, S_gain, Q_asso, h_max):
        K = S_gain.shape[0]
        S_gain_T_no_asso_no_diag = S_gain.copy().transpose().tolil()
        nz_idx_asso_x, nz_idx_asso_y = Q_asso.nonzero()
        S_gain_T_no_asso_no_diag[nz_idx_asso_x, nz_idx_asso_y] = 0
        S_gain_T_no_asso_no_diag = S_gain_T_no_asso_no_diag.tocsr()
        S_gain_T_no_asso_no_diag.setdiag(0)
        S_gain_T_no_asso_no_diag.eliminate_zeros()
        S_gain_T_no_asso_no_diag.sort_indices()
        S_sum = np.asarray(S_gain_T_no_asso_no_diag.sum(axis=1)).ravel()

        s_max = S_gain.diagonal()
        S_gain_T_no_asso_no_diag_square = S_gain_T_no_asso_no_diag.copy()
        S_gain_T_no_asso_no_diag_square.data = S_gain_T_no_asso_no_diag_square.data ** 2
        norm_H = np.sqrt(np.asarray(S_gain_T_no_asso_no_diag_square.sum(axis=1)).ravel()) * (Z-1)/(2*Z) + np.abs(1/K*h_max-1/K/Z*S_sum)


        X = cp.Variable((K,K), symmetric=True)
        S = np.asarray(S_gain_T_no_asso_no_diag.toarray())
        SX = cp.multiply(S,X)
        const_D = (cp.diag(X) == 1)
        const_F = (X[nz_idx_asso_x,nz_idx_asso_y] <= -1./(Z-1))
        const_H = (SX.sum(axis=1)*(Z-1.)/Z <= (h_max -1./Z*S_sum))
        constraints = [X>>0,const_D,const_F,const_H]

        prob = cp.Problem(cp.Minimize(0), constraints)

        return prob, X
        
class spectral_sdp_solver(sdp_solver, STATS_OBJECT):
    def __init__(self, nit=100, rank_radio=2, alpha=1.):
        sdp_solver.__init__(self, nit=nit, rank_radio=rank_radio, alpha=alpha)
    
    def run_with_state(self, bs_iteration, Z, state):
        ps_tic = self._get_tic()
        S_gain = state[0]
        K = S_gain.shape[0]
        S_gain_sym = S_gain + S_gain.transpose()
        S_gain_sym.setdiag(0)
        S_gain_sym.eliminate_zeros()
        Laplacian = scipy.sparse.csgraph.laplacian(S_gain_sym, normed=False)
        tim = self._get_tim(ps_tic)
        self._add_np_log("spectral_problem_setup",bs_iteration,np.array([Z,K,tim]))

        solving_tic = self._get_tic()
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(Laplacian, k=Z, which='LM', return_eigenvectors=True, tol=1e-5)
        normed_eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=1, keepdims=True)
        tim = self._get_tim(solving_tic)
        self._add_np_log("spectral_solve",bs_iteration,np.array([Z,K,tim]))
        return True, normed_eigvecs
