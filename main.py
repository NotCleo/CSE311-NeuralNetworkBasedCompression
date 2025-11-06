import os, gzip, pickle, time, random, heapq
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Taken from a standard Huffman Encoding Python script that I Had found online

class HuffmanNode:
    def __init__(self, sym=None, freq=0, left=None, right=None):
        self.sym, self.freq, self.left, self.right = sym, freq, left, right
    def __lt__(self, o): return self.freq < o.freq

def huffman_encode(data):
    if len(data) == 0: return b"", {}
    freq = {}
    for b in data: freq[b] = freq.get(b, 0) + 1
    heap = [HuffmanNode(k, v) for k, v in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        a, b = heapq.heappop(heap), heapq.heappop(heap)
        heapq.heappush(heap, HuffmanNode(None, a.freq + b.freq, a, b))
    root = heap[0]
    code = {}
    def build(n, prefix=""):
        if n.sym is not None: code[n.sym] = prefix or "0"
        else: build(n.left, prefix+"0"); build(n.right, prefix+"1")
    build(root)
    bits = "".join(code[b] for b in data)
    pad = (8 - len(bits) % 8) % 8
    bits += "0"*pad
    out = bytearray([pad])
    for i in range(0, len(bits), 8): out.append(int(bits[i:i+8], 2))
    return bytes(out), code

def generate_nonlinear_logs(n_rows=2500, n_cols=8, corr_strength=0.5, seed=0, filename=None):
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(n_rows, 1))
    data = []
    for i in range(n_cols):
        noise = rng.normal(scale=(1-corr_strength)*0.5, size=(n_rows,1))
        col = np.sin((i+1)*x) + np.cos(i*x*0.5) + noise
        data.append(col)
    X = np.concatenate(data, axis=1)
    X = (X - X.min())/(X.max()-X.min()+1e-9)
    if filename:
        pd.DataFrame(X, columns=[f"f{i}" for i in range(n_cols)]).to_csv(filename, index=False)
        print(f"[+] Generated {filename} (corr~{corr_strength:.2f})")
    return X
    
def build_autoencoder(input_dim, latent_dim=4):
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.Dense(32, activation="relu")(x)
    z = layers.Dense(latent_dim, activation="linear", name="latent")(x)
    y = layers.Dense(32, activation="relu")(z)
    y = layers.Dense(64, activation="relu")(y)
    out = layers.Dense(input_dim, activation="sigmoid")(y)
    ae = keras.Model(inp, out, name="autoencoder")
    encoder = keras.Model(inp, z, name="encoder")

    latent_in = keras.Input(shape=(latent_dim,))
    dec_x = latent_in
    for layer in ae.layers[-3:]:
        dec_x = layer(dec_x)
    decoder = keras.Model(latent_in, dec_x, name="decoder")

    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return ae, encoder, decoder
  
def compress_latent(Z):
    nrows, nd = Z.shape
    Zq = np.zeros_like(Z, dtype=np.uint8)
    scales = []
    for j in range(nd):
        mn, mx = Z[:, j].min(), Z[:, j].max()
        if mx - mn < 1e-9:
            scale = (mn, 1.0); Zq[:, j] = 0
        else:
            scale = (mn, mx)
            Zq[:, j] = np.round((Z[:, j]-mn)/(mx-mn)*255).astype(np.uint8)
        scales.append(scale)
    raw = Zq.tobytes()
    t0 = time.time(); comp, code = huffman_encode(raw); t1 = time.time()
    return comp, len(comp), scales, (t1 - t0)
  
def gzip_stats(path):
    raw = open(path, "rb").read()
    t0 = time.time(); gz = gzip.compress(raw, 9); t1 = time.time()
    t2 = time.time(); gun = gzip.decompress(gz); t3 = time.time()
    assert raw == gun
    return len(raw), len(gz), t1-t0, t3-t2

def reconstruction_metrics(X_true, X_pred, threshold=0.5):
    Xb_true = (X_true > 0.5).astype(int)
    Xb_pred = (X_pred > threshold).astype(int)
    acc = accuracy_score(Xb_true.flatten(), Xb_pred.flatten())
    prec = precision_score(Xb_true.flatten(), Xb_pred.flatten())
    rec = recall_score(Xb_true.flatten(), Xb_pred.flatten())
    f1 = f1_score(Xb_true.flatten(), Xb_pred.flatten())
    try:
        auc = roc_auc_score(Xb_true.flatten(), X_pred.flatten())
    except:
        auc = np.nan
    mse = mean_squared_error(X_true, X_pred)
    rmse = np.sqrt(mse)
    return dict(Accuracy=acc, Precision=prec, Recall=rec, F1=f1, AUC=auc, MSE=mse, RMSE=rmse)



# Here Starts the Main work!!!!

def run_full_experiment(workdir="learned_full_exp_metrics", n_train=20, rows_train=2500, rows_test=2500, latent_dim=4):
  # So we made a big dataset of latent dimension size of 4
    os.makedirs(workdir, exist_ok=True)

    paths = []
    for i in range(n_train):
        c = random.uniform(0, 1)
        p = os.path.join(workdir, f"train_{i:03d}_c{c:.2f}.csv")
        generate_nonlinear_logs(rows_train, 8, c, seed=i, filename=p)
        paths.append(p)

    big = np.concatenate([pd.read_csv(p).values for p in paths])
    scaler = StandardScaler().fit(big)
    Xs = scaler.transform(big)

    ae, enc, dec = build_autoencoder(Xs.shape[1], latent_dim)
    print("[*] Training autoencoder...")
    ae.fit(Xs, Xs, epochs=20, batch_size=512, verbose=1, shuffle=True)

    results = []
    for c in [0.01, 0.30, 0.75, 0.99]:
        test_path = os.path.join(workdir, f"test_c{c:.2f}.csv")
        df = pd.DataFrame(generate_nonlinear_logs(rows_test, 8, c, seed=int(c*1000)))
        df.to_csv(test_path, index=False)

        orig, gz, gz_t, gun_t = gzip_stats(test_path)
        X = df.values.astype(np.float32)
        Xs = scaler.transform(X)

        t0 = time.time(); Z = enc.predict(Xs, verbose=0); t1 = time.time()
        enc_t = t1 - t0
        comp_bytes, comp_size, scales, comp_t = compress_latent(Z)

        t2 = time.time(); Xhat_s = dec.predict(Z, verbose=0); t3 = time.time()
        dec_t = t3 - t2
        Xhat = scaler.inverse_transform(Xhat_s)

        metrics = reconstruction_metrics(X, Xhat)

        results.append({
            "corr": c, "orig": orig, "gzip": gz,
            "gzip_t": gz_t, "gunzip_t": gun_t,
            "nn_model": len(pickle.dumps(ae.get_weights())),
            "nn_latent": comp_size,
            "nn_enc": enc_t + comp_t, "nn_dec": dec_t,
            **metrics
        })

    df = pd.DataFrame(results)
    df["gzip_ratio"] = df["gzip"] / df["orig"] * 100
    df["nn_ratio"] = (df["nn_model"] + df["nn_latent"]) / df["orig"] * 100

    print(df[["corr","Accuracy","Precision","Recall","F1","MSE","RMSE"]].to_string(index=False, formatters={k:"{:.3f}".format for k in ["corr","Accuracy","Precision","Recall","F1","MSE","RMSE"]}))
    print(f"\nAvg Compression Ratios: GZIP={df['gzip_ratio'].mean():.2f}% | NN={df['nn_ratio'].mean():.2f}%")


    plt.figure(figsize=(8,4))
    barw=0.35
    idx=np.arange(len(df))
    plt.bar(idx-barw/2,df["gzip_t"],barw,label="GZIP Compress")
    plt.bar(idx+barw/2,df["nn_enc"],barw,label="NN Encode")
    plt.bar(idx-barw/2,df["gunzip_t"],barw,bottom=df["gzip_t"],label="GZIP Decompress")
    plt.bar(idx+barw/2,df["nn_dec"],barw,bottom=df["nn_enc"],label="NN Decode")
    plt.xticks(idx,[f"{c:.2f}" for c in df["corr"]])
    plt.ylabel("Seconds")
    plt.title("Encode/Decode Timing Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(workdir,"compression_time_comparison.png"))

    plt.figure(figsize=(8,5))
    barw=0.25
    plt.bar(idx-barw, df["orig"]/1024, width=barw, label="Original", color="#AAAAAA")
    plt.bar(idx, df["gzip"]/1024, width=barw, label="GZIP", color="#66BB6A")
    plt.bar(idx+barw, df["nn_latent"]/1024, width=barw, label="NN Latent", color="#42A5F5")
    plt.bar(idx+barw, df["nn_model"]/1024, width=barw, bottom=df["nn_latent"]/1024, label="NN Model", color="#1E88E5")
    plt.xticks(idx,[f"{c:.2f}" for c in df["corr"]])
    plt.ylabel("Size (KB)")
    plt.title("Compression Size Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(workdir,"compression_size_comparison.png"))

    plt.figure(figsize=(8,5))
    plt.plot(df["corr"], df["Precision"], "o-", label="Precision")
    plt.plot(df["corr"], df["Recall"], "s-", label="Recall")
    plt.plot(df["corr"], df["F1"], "^-", label="F1-Score")
    plt.plot(df["corr"], df["MSE"], "x--", label="MSE")
    plt.xlabel("Correlation Strength")
    plt.ylabel("Metric Value")
    plt.title("Reconstruction Metrics vs Correlation")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(workdir, "reconstruction_metrics_vs_corr.png"))

    print("[+] All plots saved in", workdir)
    return df

if __name__ == "__main__":
    df = run_full_experiment(workdir="learned_full_exp_metrics", n_train=30, rows_train=2500, rows_test=2500, latent_dim=4)
