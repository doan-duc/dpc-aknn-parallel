"""
demo.py - Live demo THỰC SỰ: chạy thuật toán Python từng bước, cập nhật plot real-time.
Hỗ trợ 70K mẫu: PCA 2D để vẽ nhanh, chunked delta, batch step6, subsample scatter.

Sử dụng:
    python demo.py --data <X.csv> --labels <y.csv> [--clusters 10] [--k 15] [--samples 70000]
"""
import argparse, os, sys, time
import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

COLORS = ["#FF4136","#0074D9","#2ECC40","#FF851B","#B10DC9",
          "#7FDBFF","#FFDC00","#F012BE","#01FF70","#AAAAAA"]
BG = "#0D1117"; FG = "#E6EDF3"
FASHION = ["T-shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.join(SCRIPT_DIR,"..","output","plots"); os.makedirs(PLOTS_DIR,exist_ok=True)

# ── I/O ───────────────────────────────────────────────────────────────────────
def read_int_csv(path):
    rows=[]
    with open(path,"r",encoding="utf-8",errors="replace") as f:
        for line in f:
            try: rows.append(int(line.strip().split(",")[0]))
            except ValueError: pass
    return np.array(rows,dtype=np.int32)

def read_all(data_path, label_path, n_max):
    y = read_int_csv(label_path)
    print(f"[demo] Doc du lieu X (toi da {n_max} dong)...")
    rows=[]
    with open(data_path,"r",encoding="utf-8",errors="replace") as f:
        for i,line in enumerate(f):
            if i>=n_max: break
            try: rows.append([float(v) for v in line.strip().split(",")])
            except ValueError: pass
    X = np.array(rows,dtype=np.float32)
    y = y[:len(X)]
    print(f"[demo] X={X.shape}")
    return X, y

# ── Reduce 2D ─────────────────────────────────────────────────────────────────
def reduce_2d(X, n_viz=5000):
    """PCA 2D (nhanh, hỗ trợ 70K). Chỉ vẽ n_viz điểm ngẫu nhiên."""
    print("[demo] PCA(2D) de ve...")
    pca = PCA(n_components=2, random_state=42)
    X2d = pca.fit_transform(X)
    n = len(X)
    viz_idx = np.random.RandomState(42).choice(n, min(n_viz, n), replace=False)
    print(f"[demo] Dung {len(viz_idx)} diem de ve (toan bo {n} mau duoc tinh toan)")
    return X2d, viz_idx

# ── DPC-AKNN Steps ────────────────────────────────────────────────────────────
def step1_knn(X, k):
    """[DOMAIN] sklearn KD-tree/Ball-tree."""
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm="auto", n_jobs=-1).fit(X)
    d,idx = nbrs.kneighbors(X)
    return idx[:,1:], d[:,1:]

def step2_dc(knn_dist):
    """[DOMAIN+SERIAL] d_c thich ung."""
    d_ci = knn_dist.mean(1) + knn_dist.std(1)
    return float(d_ci.mean() + d_ci.std(ddof=1))

def step3a_rho(knn_dist, d_c):
    """[DOMAIN] Gaussian kernel."""
    s = max(d_c, 1e-10)
    return np.sum(np.exp(-(knn_dist**2)/(s**2)), axis=1)

def step3b_delta_onthefly(X, rho):
    """
    [DOMAIN] Tinh delta on-the-fly — DUNG Y HET C CODE (dpc_aknn_core.c).
    Cong thuc (Eq.4): delta_i = min_{j: rho[j]>rho[i]} dist(x_i, x_j)
    Khong luu D[nxn]. Moi buoc tinh dist(X[i], X[mask]) bang numpy vector.
    Bo nho: O(n_mask * d) moi buoc ~ toi da O(n * d) = ~220MB voi 70K mau.
    """
    n = len(X)
    delta = np.zeros(n, dtype=np.float64)
    top = int(np.argmax(rho))

    # delta[top] = max dist den bat ky diem nao (on-the-fly)
    diffs_top = X.astype(np.float64) - X[top].astype(np.float64)  # (n, d)
    delta[top] = float(np.sqrt((diffs_top**2).sum(1)).max())
    max_d = delta[top]

    arange_n = np.arange(n)
    for i in range(n):
        if i == top: continue
        # Mask: diem j co rho[j] > rho[i], hoac bang rho nhung index nho hon
        mask = (rho > rho[i]) | ((rho == rho[i]) & (arange_n < i))
        if not mask.any():
            delta[i] = max_d
            continue
        # Tinh khoang cach on-the-fly chi den cac diem trong mask
        diffs = X[mask].astype(np.float64) - X[i].astype(np.float64)  # (n_mask, d)
        delta[i] = float(np.sqrt((diffs**2).sum(1)).min())
        if i % 5000 == 0:
            print(f"\r[B3b] delta: {100*i/n:.1f}%  ({i}/{n})", end="", flush=True)
    print(f"\r[B3b] delta: 100.0%  ({n}/{n})")
    return delta

def step4_centers(rho, delta, n_clusters):
    gamma = rho * delta
    return np.argsort(-gamma)[:n_clusters], gamma

def step5_initial_clusters(labels, centers, X, knn_idx, knn_dist, d_c):
    """[SERIAL] BFS."""
    n, k = knn_idx.shape
    for c, center in enumerate(centers):
        if labels[center]<0: labels[center]=c
        queue=[]
        for t in range(k):
            nb=knn_idx[center,t]
            if labels[nb]<0: labels[nb]=c
            queue.append(nb)
        mc=labels==c; centroid=X[mc].mean(0) if mc.any() else X[center].copy()
        head=0
        while head<len(queue):
            xp=queue[head]; head+=1
            lm=knn_dist[xp].mean()
            for t in range(k):
                xq=knn_idx[xp,t]
                if labels[xq]>=0: continue
                if knn_dist[xp,t]>lm: continue
                if np.linalg.norm(centroid-X[xq])>d_c: continue
                labels[xq]=c; queue.append(xq)
                mc=labels==c; centroid=X[mc].mean(0)
    return labels

def step6_association_batch(labels, knn_idx, knn_dist, rho, n_clusters):
    """
    [DOMAIN+SERIAL] Batch assignment: moi vong gan TAT CA diem co diem so duong.
    Nhanh hon phien ban cu (1 diem/vong) nhung van dung Eq.(10).
    """
    max_passes=200
    for p in range(max_passes):
        unassigned=np.where(labels<0)[0]
        if len(unassigned)==0: break
        n_u=len(unassigned); A=np.zeros((n_u,n_clusters))
        for ri,i in enumerate(unassigned):
            for t in range(knn_idx.shape[1]):
                nb=knn_idx[i,t]; cl=labels[nb]
                if cl<0: continue
                d=max(knn_dist[i,t],1e-10)
                A[ri,cl]+=(1.0/d)*rho[nb]*rho[i]
        assignable = A.max(1) > 0
        if not assignable.any(): break
        labels[unassigned[assignable]] = A[assignable].argmax(1)
        pct=100*(labels>=0).sum()/len(labels)
        print(f"\r[B6] Pass {p+1}: {pct:.1f}% duoc gan", end="", flush=True)
    print()
    return labels

def step7_voting(labels, rho, knn_idx, knn_dist, n_clusters):
    """[DOMAIN] Double-buffer."""
    order=np.argsort(-rho); new_labels=labels.copy()
    for i in order:
        lbs=labels[knn_idx[i]]; valid=lbs[lbs>=0]
        if len(valid)==0: continue
        counts=np.bincount(valid,minlength=n_clusters)
        best=counts.max()
        cands=np.where(counts==best)[0]
        if len(cands)==1: new_labels[i]=cands[0]; continue
        bm,bc=1e100,-1
        for c in cands:
            mask=labels[knn_idx[i]]==c
            if mask.any():
                m=knn_dist[i,mask].mean()
                if m<bm: bm=m; bc=c
        if bc>=0: new_labels[i]=bc
    return new_labels

def step8_remaining(labels, knn_idx, knn_dist, n_clusters):
    """[DOMAIN] Vet can."""
    for i in np.where(labels<0)[0]:
        bm,bc=1e100,0
        for c in range(n_clusters):
            mask=labels[knn_idx[i]]==c
            if mask.any():
                m=knn_dist[i,mask].mean()
                if m<bm: bm=m; bc=c
        labels[i]=bc
    return labels

def compute_metrics(yt,yp,nc):
    ari=adjusted_rand_score(yt,yp)
    nmi=normalized_mutual_info_score(yt,yp,average_method="arithmetic")
    n=max(yt.max(),yp.max())+1
    C=np.zeros((n,n),dtype=int)
    for t,p in zip(yt,yp):
        if 0<=t<n and 0<=p<n: C[t,p]+=1
    r,c=linear_sum_assignment(-C)
    acc=C[r,c].sum()/len(yt)
    return {"ARI":ari,"NMI":nmi,"ACC":acc}

# ── Live Plot ─────────────────────────────────────────────────────────────────
class LivePlot:
    def __init__(self, X2d, viz_idx, y_true, n_cls):
        self.X2d=X2d; self.vi=viz_idx; self.yt=y_true; self.nc=n_cls
        plt.ion()
        self.fig=plt.figure(figsize=(20,9),facecolor=BG)
        try: self.fig.canvas.manager.set_window_title("DPC-AKNN Live Demo")
        except: pass
        gs=gridspec.GridSpec(2,3,height_ratios=[8,1.5],width_ratios=[1,1,.5],hspace=.12,wspace=.25)
        self.axgt=self.fig.add_subplot(gs[0,0])
        self.axpd=self.fig.add_subplot(gs[0,1])
        self.axif=self.fig.add_subplot(gs[0,2])
        self.axst=self.fig.add_subplot(gs[1,:])
        for ax in [self.axgt,self.axpd,self.axif,self.axst]:
            ax.set_facecolor(BG)
            for sp in ax.spines.values(): sp.set_edgecolor("#30363D")
        self.axif.axis("off"); self.axst.axis("off")
        self.fig.suptitle("DPC-AKNN  |  CPU Parallel (OpenMP)  |  Fashion-MNIST  |  Live Demo",
                           color=FG,fontsize=13,fontweight="bold")
        self._draw_gt(); plt.pause(0.01)

    def _sc(self, ax, labels_full, title, names=None):
        ax.cla(); ax.set_facecolor(BG)
        ax.set_title(title,color=FG,fontsize=10,fontweight="bold")
        ax.tick_params(colors="#555",labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363D")
        lv=labels_full[self.vi]; Xv=self.X2d[self.vi]
        mask_un=lv<0
        if mask_un.any():
            ax.scatter(Xv[mask_un,0],Xv[mask_un,1],c="#2a2a2a",s=3,alpha=.5,linewidths=0)
        for c in range(self.nc):
            m=lv==c
            if m.any(): ax.scatter(Xv[m,0],Xv[m,1],c=COLORS[c%10],s=5,alpha=.8,linewidths=0)
        ns=names or [f"Cum{i}" for i in range(self.nc)]
        patches=[mpatches.Patch(color=COLORS[i%10],label=ns[i] if i<len(ns) else f"C{i}") for i in range(self.nc)]
        ax.legend(handles=patches,loc="lower left",fontsize=6.5,framealpha=.3,
                  labelcolor="white",facecolor="#161B22",edgecolor="#30363D",ncol=2)

    def _draw_gt(self): self._sc(self.axgt,self.yt,"Ground-Truth  (Nhan thuc te)",FASHION)

    def update(self, labels_full, desc, metrics=None):
        n=len(labels_full); na=(labels_full>=0).sum()
        self._sc(self.axpd,labels_full,f"DPC-AKNN Prediction  [{na}/{n} da gan  ({100*na/n:.1f}%)]")
        self.axst.cla(); self.axst.axis("off"); self.axst.set_facecolor("#161B22")
        col="#2ECC40" if "DONE" in desc or "HOAN" in desc else "#58A6FF"
        self.axst.text(.01,.5,desc,color=col,fontsize=9,transform=self.axst.transAxes,
                       va="center",fontfamily="monospace")
        self.axif.cla(); self.axif.axis("off"); self.axif.set_facecolor(BG)
        if metrics:
            self.axif.text(.05,.97,">> Danh gia:",color=FG,fontsize=11,fontweight="bold",
                           transform=self.axif.transAxes,va="top",fontfamily="monospace")
            yp=.78
            for nm,v in metrics.items():
                c="#2ECC40" if v>.6 else("#FF851B" if v>.4 else"#FF4136")
                bar="█"*int(v*12)+"░"*(12-int(v*12))
                self.axif.text(.05,yp,f"{nm}: {v:.4f}\n[{bar}]",color=c,fontsize=9,
                               transform=self.axif.transAxes,va="top",fontfamily="monospace")
                yp-=.22
        plt.tight_layout(rect=[0,0,1,.95]); self.fig.canvas.draw(); plt.pause(0.05)

    def finish(self):
        ts=time.strftime("%Y%m%d_%H%M%S")
        path=os.path.join(PLOTS_DIR,f"demo_{ts}.png")
        self.fig.savefig(path,dpi=120,bbox_inches="tight",facecolor=BG)
        print(f"[demo] Luu: {path}")
        plt.ioff(); plt.show()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--labels",required=True)
    ap.add_argument("--clusters",type=int,default=10)
    ap.add_argument("--k",type=int,default=15)
    ap.add_argument("--samples",type=int,default=70000)
    ap.add_argument("--viz",type=int,default=5000,help="So diem hien thi tren plot (mac dinh 5000)")
    ap.add_argument("--chunk",type=int,default=500,help="Chunk size cho delta (mac dinh 500)")
    args=ap.parse_args()
    K=args.k; NC=args.clusters

    X, y_true = read_all(args.data, args.labels, args.samples)
    n=len(X)

    X2d, viz_idx = reduce_2d(X, args.viz)
    live=LivePlot(X2d, viz_idx, y_true, NC)
    labels=np.full(n,-1,dtype=int)
    t0_all=time.time()

    def show(desc, lbl=None, met=None):
        live.update(lbl if lbl is not None else labels, desc, met)

    # B1
    show("Buoc 1/8: Tinh kNN [DOMAIN] sklearn KD-tree O(n log n)...")
    t0=time.time(); knn_idx,knn_dist=step1_knn(X,K)
    print(f"[B1] {time.time()-t0:.1f}s")
    show("Buoc 1/8: kNN DONE [DOMAIN] — O(n log n), khong can D[nxn]")

    # B2
    show("Buoc 2/8: Tinh d_c [DOMAIN+SERIAL]...")
    d_c=step2_dc(knn_dist)
    print(f"[B2] d_c={d_c:.6f}")
    show(f"Buoc 2/8: d_c={d_c:.6f}  DONE")

    # B3a
    show("Buoc 3/8a: Tinh rho [DOMAIN] Gaussian kernel...")
    t0=time.time(); rho=step3a_rho(knn_dist,d_c)
    print(f"[B3a] {time.time()-t0:.1f}s")
    show("Buoc 3/8a: Rho DONE [DOMAIN] — chi dung knn_dist, khong can D[nxn]")

    # B3b
    show("Buoc 3/8b: Tinh delta [DOMAIN chunked] — on-the-fly, O(n^2) time, O(chunk*n) memory...")
    t0=time.time()
    delta=step3b_delta_onthefly(X, rho)
    print(f"[B3b] {time.time()-t0:.1f}s")
    show("Buoc 3/8b: Delta DONE [DOMAIN] — chunked, khong luu D[nxn]")

    # B4
    centers,gamma=step4_centers(rho,delta,NC)
    lbl4=np.full(n,-1,dtype=int)
    for ci,c in enumerate(centers): lbl4[c]=ci
    show("Buoc 4/8: Chon tam cum [SERIAL] gamma=rho*delta top-k", lbl4)
    labels=lbl4.copy()

    # B5
    show("Buoc 5/8: Xay dung cum ban dau [SERIAL] BFS 3 dieu kien...")
    t0=time.time()
    labels=step5_initial_clusters(labels,centers,X,knn_idx,knn_dist,d_c)
    na=(labels>=0).sum()
    print(f"[B5] {na}/{n} ({time.time()-t0:.1f}s)")
    show("Buoc 5/8: Cum nong cot DONE [SERIAL] — BFS admission")

    # B6
    show("Buoc 6/8: Ma tran lien ket A [DOMAIN batch]...")
    t0=time.time()
    labels=step6_association_batch(labels,knn_idx,knn_dist,rho,NC)
    na=(labels>=0).sum()
    print(f"[B6] {na}/{n} ({time.time()-t0:.1f}s)")
    show("Buoc 6/8: Association DONE [DOMAIN batch] — nhanh hon 1-diem/vong")

    # B7
    show("Buoc 7/8: Bau chon sua loi [DOMAIN double-buffer]...")
    t0=time.time(); labels=step7_voting(labels,rho,knn_idx,knn_dist,NC)
    print(f"[B7] {time.time()-t0:.1f}s")
    show("Buoc 7/8: Voting DONE [DOMAIN] — thu tu rho giam dan")

    # B8
    show("Buoc 8/8: Vet can ngoai lai [DOMAIN]...")
    t0=time.time(); labels=step8_remaining(labels,knn_idx,knn_dist,NC)
    print(f"[B8] {time.time()-t0:.1f}s")

    metrics=compute_metrics(y_true,labels,NC)
    total=time.time()-t0_all
    print(f"\n{'='*50}")
    print(f"  KET QUA - DPC-AKNN  |  n={n}  |  k={K}  |  clusters={NC}")
    print(f"{'='*50}")
    for nm,v in metrics.items():
        bar="█"*int(v*20)+"░"*(20-int(v*20))
        print(f"  {nm}: {v:.4f}  [{bar}]")
    print(f"  Tong thoi gian: {total:.1f}s")
    print(f"{'='*50}\n")

    show(">> HOAN THANH! n={} | {:.1f}s".format(n,total), met=metrics)
    live.update(labels,">> HOAN THANH - DPC-AKNN da phan loai xong 100% du lieu!",metrics)
    live.finish()

if __name__=="__main__":
    main()
