import streamlit as st
import json
import os
import tempfile
import networkx as nx
from pyvis.network import Network
from matplotlib import colors as mcolors
from math import cos, sin, pi
import numpy as np

data_by_year = {
    2021: {
        "nodes": [
            ("TEO|Professor", {"fac": "TEO", "grp": "Professor", "size": 60}),
            ("TEO|Lektor",    {"fac": "TEO", "grp": "Lektor",    "size": 35}),
            ("TEO|Ph.d.",    {"fac": "TEO", "grp": "Ph.d.",    "size": 35}),
            ("TEO|Adjunkt",    {"fac": "TEO", "grp": "Adjunkt",    "size": 35}),
            ("HUM|Ph.d.",     {"fac": "HUM", "grp": "Ph.d.",     "size": 22}),
            ("HUM|Lektor",    {"fac": "HUM", "grp": "Lektor",    "size": 28}),
            ("SCIENCE|Postdoc", {"fac":"SCIENCE","grp":"Postdoc","size": 25}),
        ],
        "edges": [
            ("TEO|Professor", "HUM|Ph.d.",   2.0), # inter
            ("TEO|Lektor",    "HUM|Lektor",  1.0), # inter
            ("TEO|Professor", "TEO|Lektor",  4.6), # intra
            ("TEO|Professor", "TEO|Adjunkt",  0.6), # intra
            ("TEO|Lektor",    "TEO|Ph.d.",  1.6), # intra
            ("HUM|Ph.d.",     "SCIENCE|Postdoc", 1.5), # inter
            ("HUM|Lektor",    "HUM|Ph.d.",  1.6), # intra
        ]
    },
    2022: {
        "nodes": [
            ("TEO|Professor",   {"fac":"TEO","grp":"Professor","size": 52}),
            ("TEO|Adjunkt",     {"fac":"TEO","grp":"Adjunkt",  "size": 18}),
            ("HUM|Ph.d.",       {"fac":"HUM","grp":"Ph.d.",    "size": 30}),
            ("SCIENCE|Postdoc", {"fac":"SCIENCE","grp":"Postdoc","size": 40}),
            ("SCIENCE|Professor", {"fac":"SCIENCE","grp":"Professor","size": 55}),
        ],
        "edges": [
            ("TEO|Professor",   "HUM|Ph.d.",        2.5),
            ("SCIENCE|Postdoc", "SCIENCE|Professor",1.2),
            ("TEO|Adjunkt",     "SCIENCE|Postdoc",  0.8),
        ]
    }
}

hierarki = {
    "Ph.d.": 1,
    "Stillinger u. adjunktniveau": 2,
    "Postdoc": 3,
    "Adjunkt": 4,
    "Lektor": 5,
    "Professor": 6
}
group_order = sorted(hierarki.keys(), key=lambda g: hierarki[g])
lvl_min, lvl_max = min(hierarki.values()), max(hierarki.values())

fac_order = ["SAMF", "JUR", "TEO", "SUND", "HUM", "SCIENCE"]

# Layout for alle år
pos = {"A": (0,0), "B": (1,1), "C": (2,0)}

with open(r"ku-farver02.json", encoding="utf-8") as f:
    ku_farver = json.load(f)

faculty_base_colors = {
    "TEO": ku_farver["Roed"]["Lys"],
    "JUR": ku_farver["HvidGul"]["Gul"],
    "HUM": ku_farver["Blaa"]["Moerk"],
    "SCIENCE": ku_farver["Groen"]["Moerk"],
    "SAMF": ku_farver["Petroleum"]["Moerk"],
    "SUND": ku_farver["Roed"]["Moerk"],
}


def lighten_color(hex_color, amount):
    """ Lysner hex-farve med faktor [0..1] """
    try:
        c = np.array(mcolors.to_rgb(hex_color))
    except ValueError:
        c = np.array([0.6, 0.6, 0.6])
    white = np.array([1, 1, 1])
    mix = (1 - amount) * c + amount * white
    return mcolors.to_hex(mix)

def add_alpha(hex_color, alpha):
    r, g, b = mcolors.to_rgb(hex_color)
    return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {alpha})"


# UI: filtre
st.set_page_config(layout = "wide")
st.title("KU netværk")

years = list(data_by_year.keys())

# Alle fakulteter, der forekommer i datasæt
all_faculties = sorted({meta["fac"] for content in data_by_year.values() for _, meta in content["nodes"]})
all_groups = sorted({meta["grp"] for content in data_by_year.values() for _, meta in content["nodes"]}, key = lambda g: hierarki.get(g, 999))

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    year = st.selectbox("Vægt år: ", list(data_by_year.keys()), index = 0)

    # Sæt slider max ud fra pågældende års data
    raw_weights = [w for _, _, w in data_by_year[year]["edges"]]
    max_w = max(raw_weights) if raw_weights else 1.0
    min_w = st.slider("Min. kantvægt", 0.0, float(max_w), 0.0, 0.1)

with col2:
    selected_facs = st.multiselect("Fakulteter (tom = alle): ", all_faculties, default = [])
    selected_grps = st.multiselect("Stillingsgrupper (tom = alle): ", all_groups, default = [])

    show_intra = st.checkbox("Vis intra-fakultetskanter", True)
    show_inter = st.checkbox("Vis inter-fakultetskanter", True)

with col3:
    node_min_px = st.slider("Node min px", 6, 30, 10)
    node_max_px = st.slider("Node max px", 30, 100, 60)
    edge_scale = st.slider("Kant-skala", 1.0, 15.0, 6.0, 0.5)

# Byg grafen for valgte år + filtre
content = data_by_year[year]
node_meta = {nid: meta for nid, meta in content["nodes"]}

# Nodefilter
if selected_facs:
    keep_by_fac = {nid for nid, m in content["nodes"] if m["fac"] in selected_facs}
else:
    keep_by_fac = set(node_meta.keys())

if selected_grps:
    keep_by_grp = {nid for nid, m in content["nodes"] if m["grp"] in selected_grps}
else:
    keep_by_grp = set(node_meta.keys())

nodes_keep = keep_by_fac & keep_by_grp

def edge_type(u, v):
    """ Returnerer "intra", hvis samme fakultet, ellers "inter"""
    fu = node_meta[u]["fac"]
    fv = node_meta[v]["fac"]
    return "intra" if fu == fv else "inter"

edges_keep = []
for u, v, w in content["edges"]:
    if u not in nodes_keep or v not in nodes_keep:
        continue
    et = edge_type(u, v)
    if et == "intra" and not show_intra:
        continue
    if et == "inter" and not show_inter:
        continue
    if w < min_w:
        continue
    edges_keep.append((u, v, w))


# Layout: fakulteter på stor cirkel, grupper på lille cirkel per fakultet
fac_present = [fac for fac in fac_order if fac in all_faculties]
n_fac = max(1, len(fac_present))
R_fac = 600
r_grp = 180

fac_angle = {fac: 2 * pi * i / n_fac for i, fac in enumerate(fac_present)}
fac_center = {fac: (R_fac * cos(fac_angle[fac]), R_fac * sin(fac_angle[fac])) for fac in fac_present}

# grupper per fakultet
groups_by_fac = {
    fac: sorted({node_meta[n]["grp"] for n in nodes_keep if node_meta[n]["fac"] == fac}, 
    key = lambda g: group_order.index(g) if g in group_order else 999)
    for fac in fac_present
}

# Position for hvert node-id
pos = {}
for fac in fac_present:
    cx, cy = fac_center[fac]
    grps = groups_by_fac[fac]
    k = max(1, len(grps))
    for i, g in enumerate(grps):
        theta = 2 * pi * i / k
        gx = cx + r_grp * cos(theta)
        gy = cy + r_grp * sin(theta)
        
        for nid in [n for n in nodes_keep if node_meta[n]["fac"] == fac and node_meta[n]["grp"] == g]:
            pos[nid] = (gx, gy)

# Skaler node-størrelser (til pixels)
raw_sizes = [node_meta[n].get("size", 1) for n in nodes_keep] or [1]
s_min, s_max = min(raw_sizes), max(raw_sizes)
def scale_size(val, vmin = s_min, vmax = s_max, px_min = node_min_px, px_max = node_max_px):
    if vmax <= vmin:
        return (px_min + px_max) / 2
    t = (val - vmin) / (vmax - vmin)
    return px_min + t*(px_max - px_min)


# Netværksmetrik
G = nx.Graph()
G.add_nodes_from(nodes_keep)
for u, v, w in edges_keep:
    G.add_edge(u, v, weight = w)

density = nx.density(G)

# Fakultetsbaseret modularitet
communities = []
for fac in all_faculties:
    comm = [n for n in nodes_keep if node_meta[n]["fac"] == fac]
    if comm:
        communities.append(comm)

if communities and G.number_of_edges() > 0:
    from networkx.algorithms.community.quality import modularity
    modularity_fac = modularity(G, communities, weight = "weight")
else:
    modularity_fac = float("nan")

# Greedy modularitet
if G.number_of_edges() > 0 and G.number_of_nodes() > 1:
    from networkx.algorithms.community import greedy_modularity_communities
    from networkx.algorithms.community.quality import modularity as modq

    greedy_comms = list(greedy_modularity_communities(G, weight = "weight"))
    modularity_greedy = modq(G, greedy_comms, weight = "weight")
    n_comms = len(greedy_comms)
else:
    greedy_comms = []
    modularity_greedy = float("nan")
    n_comms = 0

# weighted degree centrality (strength)
weighted_deg = {node: sum(data["weight"] for _, _, data in G.edges(node, data = True)) for node in G.nodes}

faculty_wd = {}
for node, val in weighted_deg.items():
    fac = node_meta[node]["fac"]
    faculty_wd.setdefault(fac, 0)
    faculty_wd[fac] += val
faculty_wd_sorted = sorted(faculty_wd.items(), key=lambda x: -x[1])

# betweenness centrality
bet_cent = nx.betweenness_centrality(G, weight = "weight", normalized = True)

faculty_bs = {}
for node, val in bet_cent.items():
    fac = node_meta[node]["fac"]
    faculty_bs.setdefault(fac, 0)
    faculty_bs[fac] += val
faculty_bs_sorted = sorted(faculty_bs.items(), key=lambda x: -x[1])

# Opret PyVis-netværket
net = Network(height = "700px", width = "100%", directed = False)
net.toggle_physics(False)

for nid in nodes_keep:
    fac = node_meta[nid]["fac"]
    grp = node_meta[nid]["grp"]
    base = faculty_base_colors.get(fac, "black")
    lvl = hierarki.get(grp, lvl_min)
    amt = 0.65 - 0.55 * ((lvl - lvl_min) / (lvl_max - lvl_min))
    color = lighten_color(base, amt)

    x, y = pos[nid]
    size_px = scale_size(node_meta[nid].get("size", 1))

    net.add_node(
        nid,
        label = f"{grp}",
        x = x, y = y,
        size = size_px,
        color = color,
        title = f"{fac} - {grp} \n Forfatterbidrag: {node_meta[nid].get('size', 'NA')}",
        physics = False
    )

if edges_keep:
    max_w_cur = max(w for _, _, w in edges_keep)
else:
    max_w_cur = 1.0

for u, v, w in edges_keep:
    width = 10 + edge_scale * (w / max_w_cur)
    et = "intra" if node_meta[u]["fac"] == node_meta[v]["fac"] else "inter"
    col = "black" if et == "inter" else lighten_color(faculty_base_colors[node_meta[u]["fac"]], 0.25)

    net.add_edge(
        u, v, 
        value = w,
        width = width,
        color = add_alpha(col, 0.4),
        title = f"Vægt: {w:.2f} ({et})"
    )






# Byg graf
temp_path = os.path.join(tempfile.gettempdir(), f"pyvis_{year}.html")
net.write_html(temp_path)
with open(temp_path, "r", encoding="utf-8") as f:
    html = f.read()
st.components.v1.html(html, height=800, scrolling=True)

st.subheader("Netværksmetrik for valgte udsnit")
#st.table(metrics)


c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Nodes", len(G.nodes),
              help="Antal noder i netværket efter filtre")

with c2:
    st.metric("Edges", len(G.edges),
              help="Antal forbindelser mellem noderne i netværket")

with c3:
    st.metric("Densitet", round(density, 4),
              help="Densitet er et mål for, hvor stor en andel af alle mulige forbindelser i netværket, der faktisk findes")

c4, c5, c6 = st.columns(3)
with c4:
    st.metric("Modularitet", round(modularity_fac, 4),
              help="Modularitet baseret på KU-fakulteter som klynger. Modularitet er et mål for, hvor stærkt netværket er opdelt i adskilte klynger sammenlignet med, hvad man ville forvente tilfældigt")

with c5:
    st.metric("Greedy modularitet", round(modularity_greedy, 4),
              help="Modularitet beregnet med greedy community detection; uden forudbestemte klynger.")

with c6:
    st.metric("Antal greedy-klynger", n_comms,
              help="Klynger beregnet med greedy community detection.")

st.subheader("Centralitet i valgte udsnit")
c11, c12 = st.columns(2)
with c11:
    top_fac, top_fac_val = faculty_wd_sorted[0]
    st.metric(
        "Fakultet med højest weighted degree",
        top_fac,
        help = f"Samlet strength for fakultetet: {top_fac_val:.3f}"
    )

with c12:
    top_fac, top_fac_val = faculty_wd_sorted[-1]
    st.metric(
        "Fakultet med lavest weighted degree",
        top_fac,
        help = f"Samlet strength for fakultetet: {top_fac_val:.3f}"
    )

c13, c14 = st.columns(2)
with c13:
    top_fac, top_fac_val = faculty_bs_sorted[0]
    st.metric(
        "Fakultet med højest brobyggerfaktor",
        top_fac,
        help = f"Samlet betweenness for fakultetet: {top_fac_val:.3f}"
    )

with c14:
    top_fac, top_fac_val = faculty_bs_sorted[-1]
    st.metric(
        "Fakultet med lavest brobyggerfaktor",
        top_fac,
        help = f"Samlet betweenness for fakultetet: {top_fac_val:.3f}"
    )


st.subheader("Centralitet for stillingsgrupper i valgte udsnit")
# Sortér efter højeste weighted degree
wd_sorted = sorted(weighted_deg.items(), key=lambda x: -x[1])

# Sortér betweenness
bc_sorted = sorted(bet_cent.items(), key=lambda x: -x[1])

c7, c8 = st.columns(2)
with c7:
    if wd_sorted:
        top_node, top_val = wd_sorted[0]
        fac = node_meta[top_node]["fac"]
        grp = node_meta[top_node]["grp"]
        st.metric(
            f"Stillingsgruppe med højest weighted degree",
            grp,
            help = "hej"
        )

with c8:
    if wd_sorted:
        top_node, top_val = wd_sorted[-1]
        fac = node_meta[top_node]["fac"]
        grp = node_meta[top_node]["grp"]
        st.metric(
            f"Stillingsgruppe med lavest weighted degree",
            grp,
            help = "hej"
        )

c9, c10 = st.columns(2)
with c9:
    if bc_sorted:
        top_node, top_val = wd_sorted[0]
        fac = node_meta[top_node]["fac"]
        grp = node_meta[top_node]["grp"]
        st.metric(
            f"Stillingsgruppe med højest brobyggerfaktor",
            grp,
            help = "hej"
        )

with c10:
    if bc_sorted:
        top_node, top_val = wd_sorted[-1]
        fac = node_meta[top_node]["fac"]
        grp = node_meta[top_node]["grp"]
        st.metric(
            f"Stillingsgruppe med lavest brobyggerfaktor",
            grp,
            help = "hej"
        )





#net.write_html(r"H:/Sampubliceringsanalyse/PyVis_test/simple_graph.html")
