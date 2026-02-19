import streamlit as st
import json
import os
import tempfile
import networkx as nx
from pyvis.network import Network
from matplotlib import colors as mcolors
from math import cos, sin, pi
import numpy as np
import colorsys
import pyarrow as pa
import io, csv
import datetime
from collections import defaultdict
import plotly.graph_objects as go
from networkx.algorithms.community.quality import modularity as modq
from networkx.algorithms.community import greedy_modularity_communities
import ast
import math


def scale_size_log(val, px_min=5, px_max=60):
    if val <= 0:
        return px_min
    log_val = math.log1p(val)
    # normaliser til [px_min, px_max]
    return px_min + (log_val / math.log1p(global_max_auth)) * (px_max - px_min)


def build_table(rows, schema_fields):
    """ rows list[dict] med keys """

    cols = {}
    for name, dtype in schema_fields:
        cols[name] = [r.get(name) for r in rows]
    arrays = [pa.array(cols[name], type = dtype) for name, dtype in schema_fields]
    return pa.Table.from_arrays(arrays, names = [name for name, _ in schema_fields])

def rows_to_csv_bytes(rows, field_order):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames = field_order, extrasaction = "ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k) for k in field_order})
    return buf.getvalue().encode("utf-8")



json_path = "H:/Sampubliceringsanalyse/Data/vip_transformed.json"

with open(json_path, "r", encoding="utf-8") as f:
    data_by_year = json.load(f)

# Konverter keys fra str → int (json gemmer nøgler som strings)
data_by_year = {int(year): val for year, val in data_by_year.items()}






institut_fakultets_map = {}

for year_content in data_by_year.values():
    for nid, meta in year_content["nodes"]:
        fac = meta.get("fac")
        inst = meta.get("inst")
        if fac and inst:
            institut_fakultets_map[inst] = fac





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

with open(r"C:/Users/rjp530/Desktop/ku-farver02.json", encoding="utf-8") as f:
    ku_farver = json.load(f)

faculty_base_colors = {
    "TEO": "#3A1A5F",
    "JUR": "#7d5402",
    "HUM": ku_farver["Blaa"]["Moerk"],
    "SCIENCE": ku_farver["Groen"]["Moerk"],
    "SAMF": ku_farver["Petroleum"]["Moerk"],
    "SUND": "#7A131A",
}

def adjust_color(hex_color, lightness_factor = 1.0, saturation_factor = 1.0):
    r, g, b = mcolors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, min(1, l * lightness_factor))
    s = max(0, min(1, s * saturation_factor))

    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return mcolors.to_hex((r2, g2, b2))

def add_alpha(hex_color, alpha):
    r, g, b = mcolors.to_rgb(hex_color)
    return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {alpha})"

# UI: filtre
st.set_page_config(
    page_title = "KU Analyse",
    page_icon = "S:/FAD-REKSEK-REKSTAB-ABI-Analyse/quarto/ku_skabeloner/KU-logo.png",
    layout = "wide",
    )

col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.image("S:/FAD-REKSEK-REKSTAB-ABI-Analyse/quarto/ku_skabeloner/KU-logo.png")
with col_title:
    st.title("Sampublicering på Københavns Universitet")

#st.title("Sampublicering på Københavns Universitet")
#st.markdown('[Læs dokumentationen (PDF)]("file:///H:/Sampubliceringsanalyse/PyVis_test/dokumentation.pdf")')

years = list(data_by_year.keys())

# Alle fakulteter, der forekommer i datasæt
all_faculties = sorted({meta["fac"] for content in data_by_year.values() for _, meta in content["nodes"]})
all_groups = sorted({meta["grp"] for content in data_by_year.values() for _, meta in content["nodes"]}, key = lambda g: hierarki.get(g, 999))

# Forfatterbidrag
global_sizes = [meta["size"] for year_content in data_by_year.values() for _, meta in year_content["nodes"]]
global_min_auth = min(global_sizes) if global_sizes else 0
global_max_auth = max(global_sizes) if global_sizes else 0

# Sampublicering
all_edge_counts = [int(round(w)) for year_content in data_by_year.values() for (_, _, w) in year_content["edges"]]
global_min_w = min(all_edge_counts) if all_edge_counts else 0.0
global_max_w = max(all_edge_counts) if all_edge_counts else 1.0

cols = st.columns(2)

with cols[0]:
    year = st.selectbox("Vælg år: ", list(data_by_year.keys()), index = 0)
    show_intra = st.checkbox("Vis intra-fakultetskanter", True)
    show_inter = st.checkbox("Vis inter-fakultetskanter", True)

# Byg grafen for valgte år + filtre (FAC|INST|GRP)
content = data_by_year[year]

node_meta = {}
for nid, meta in content["nodes"]:
    m = dict(meta)
    m.setdefault("type", "grp")

    parts = nid.split("|")
    if "fac" not in m and len(parts) >= 1:
        m["fac"] = parts[0]
    if "inst" not in m and len(parts) >= 2:
        m["inst"] = parts[1]
    if "grp" not in m and len(parts) >= 3:
        m["grp"] = parts[2]

    # mangler size?
    m.setdefault("size", 0)

    node_meta[nid] = m


for nid, m in list(node_meta.items()):
    fac = m["fac"]
    inst = m["inst"]

    inst_id = f"INST:{fac}|{inst}"

    if inst_id not in node_meta:
        node_meta[inst_id] = {
            "type": "inst",
            "fac": fac,
            "inst": inst,
            "grp": "",
            "size": 0
        }


year_sizes = [meta.get("size", 0) for _, meta in content["nodes"]]

# Nodefilter i sidebar
st.sidebar.header("Hvad skal vises i netværket?")

show_fac = st.sidebar.checkbox("Fakulteter", value = True)
show_inst = st.sidebar.checkbox("Institutter", value = False)
show_grp = st.sidebar.checkbox("Stillingsgrupper", value = True)

mode = (
    ("F" if show_fac else "") +
    ("I" if show_inst else "") + 
    ("G" if show_grp else "")
)


if not show_fac and not show_inst and not show_grp:
    st.error("Vælg mindst ét filter (fakultet, institut eller stillingsgruppe).")
    st.stop()


# Alle mulige institutter
all_inst = sorted({m.get("inst", "UKENDT") for m in node_meta.values()})

if show_fac:
    selected_facs = st.sidebar.multiselect("Vælg fakulteter (tom = alle)", all_faculties, default = [])
else:
    selected_facs = []

if show_inst:
    selected_insts = st.sidebar.multiselect("Vælg institutter", all_inst, default = [])
else:
    selected_insts = []

if show_grp:
    selected_grps = st.sidebar.multiselect("Vælg stillingsgrupper", all_groups, default = [])
else:
    selected_grps = []

### Universal merge-system
# Case 1: show_grp = True,  show_inst = True    → ingen merge
# Case 2: show_grp = True,  show_inst = False   → merge GRP → (FAC, GRP)
# Case 3: show_grp = False, show_inst = True    → merge GRP → INST
# Case 4: show_grp = False, show_inst = False   → merge ALT → FAC

# Rå noder og kanter fra datasættet
raw_nodes = dict(node_meta)
raw_edges = list(content["edges"])

def merge_grp_to_facgrp(nodes, edges):
    merged_meta = {}
    old_to_new = {}
    size_acc = {}

    # Merge nodes
    for nid, m in nodes.items():
        if m.get("type") != "grp":
            continue
        
        fac = m["fac"]
        grp = m["grp"]
        new_id = f"{fac}|{grp}"

        old_to_new[nid] = new_id

        if new_id not in merged_meta:
            merged_meta[new_id] = {
                "type": "grp",
                "fac": fac,
                "inst": "",
                "grp": grp,
                "size": 0,
                "children": []
                }
            size_acc[(fac, grp)] = 0
        
        size_acc[(fac, grp)] += m.get("size", 0)
        merged_meta[new_id]["children"].append(nid)

    # Behold fakultets-noder
    for nid, m in nodes.items():
        if m.get("type") == "fac":
            merged_meta[nid] = m
    
    # Opdater sizes
    for (fac, grp), s in size_acc.items():
        new_id = f"{fac}|{grp}"
        merged_meta[new_id]["size"] = s
    
    # Merge edges
    edge_acc = {}
    for u, v, w in edges:
        if u not in old_to_new or v not in old_to_new:
            continue
        u2 = old_to_new[u]
        v2 = old_to_new[v]

        if u2 == v2:
            continue
        key = tuple(sorted((u2, v2)))
        edge_acc[key] = edge_acc.get(key, 0) + w
    
    merged_edges = [(u, v, w) for (u, v), w in edge_acc.items()]
    
    return merged_meta, merged_edges


def merge_grp_to_inst(nodes, edges):
    merged_meta = {}
    old_to_new = {}

    # 1) Først: aggreger alle GRP -> INST
    for nid, m in nodes.items():
        t = m.get("type")
        if t == "grp":
            fac = m["fac"]
            inst = m["inst"]
            new_id = f"INST:{fac}|{inst}"
            old_to_new[nid] = new_id

            if new_id not in merged_meta:
                merged_meta[new_id] = {
                    "type": "inst",
                    "fac": fac,
                    "inst": inst,
                    "grp": "",
                    "size": 0,
                    "children": []
                }

            merged_meta[new_id]["size"] += int(m.get("size", 0))
            merged_meta[new_id]["children"].append(nid)

    # 2) Dernæst: kopier evt. inst-pladsholdere KUN hvis de ikke findes
    for nid, m in nodes.items():
        t = m.get("type")
        if t == "inst":
            if nid not in merged_meta:      # ← NØGLEN: ingen overskrivning
                merged_meta[nid] = dict(m)
                merged_meta[nid].setdefault("children", [])
            old_to_new[nid] = nid
        elif t == "fac":
            # Hvis du en dag har 'fac' i 'nodes', kan du vælge at beholde dem:
            merged_meta[nid] = m
            old_to_new[nid] = nid

    # Merge edges på inst-niveau
    edge_acc = {}
    for u, v, w in edges:
        if u in old_to_new and v in old_to_new:
            u2, v2 = old_to_new[u], old_to_new[v]
            if u2 != v2:
                key = tuple(sorted((u2, v2)))
                edge_acc[key] = edge_acc.get(key, 0) + w

    merged_edges = [(u, v, w) for (u, v), w in edge_acc.items()]
    return merged_meta, merged_edges



def merge_all_to_fac(nodes, edges):

    merged_meta = {}
    old_to_new = {}

    # merge nodes
    for nid, m in nodes.items():
        fac = m["fac"]
        new_id = f"FAC:{fac}"
        old_to_new[nid] = new_id

        if new_id not in merged_meta:
            merged_meta[new_id] = {
                "type": "fac",
                "fac": fac,
                "inst": "",
                "grp": "",
                "size": 0,
                "children": []
            }

        merged_meta[new_id]["size"] += m.get("size", 0)
        merged_meta[new_id]["children"].append(nid)

    # merge edges
    edge_acc = {}
    for u, v, w in edges:
        if u not in old_to_new or v not in old_to_new:
            continue

        u2 = old_to_new[u]
        v2 = old_to_new[v]

        if u2 == v2:
            continue

        key = tuple(sorted((u2, v2)))
        edge_acc[key] = edge_acc.get(key, 0) + w

    merged_edges = [(u, v, w) for (u, v), w in edge_acc.items()]
    
    return merged_meta, merged_edges

def edge_within_weight_range(w):
    int_weight = int(round(w))
    return edge_min <= int_weight <= edge_max

def passes_category_on_raw(m):
    fac = m.get("fac", "")
    inst = m.get("inst", "")
    grp = m.get("grp", "")
    t   = m.get("type", "grp")

    # fakulteter
    if show_fac and selected_facs:
        if fac not in selected_facs:
            return False

    # institutter
    if show_inst and selected_insts:
        if inst not in selected_insts:
            return False

    # stillingsgrupper
    if show_grp and selected_grps:
        if t == "grp":
            if grp not in selected_grps:
                return False
    return True

if mode == "F":
    node_meta, edge_source = merge_all_to_fac(raw_nodes, raw_edges)

# FG = grupper merged til FAC|GRP
elif mode == "FG":
    node_meta, edge_source = merge_grp_to_facgrp(raw_nodes, raw_edges)

# FI = grupper merged til INST, beholder institutes
elif mode == "FI":
    node_meta, edge_source = merge_grp_to_inst(raw_nodes, raw_edges)

# FIG = ingen merge → originale inst + grp
elif mode == "FIG":
    node_meta, edge_source = raw_nodes, raw_edges
    # drop inst-noder helt
    node_meta = {nid: m for nid, m in node_meta.items() if m.get("type") != "inst"}


# IG = grupper merged til INST, ingen fakulteter
elif mode == "IG":
    # Merge GRP → INST (som i FI)
    nm, es_inst_edges = merge_grp_to_inst(raw_nodes, raw_edges)

    # 2) Behold INST fra merge (med children) + tilføj ALLE originale GRP-noder (fra raw_nodes)
    node_meta = {}

    # INST fra merge (har children, size=summen)
    for nid, m in nm.items():
        if m.get("type") == "inst":
            node_meta[nid] = m

    # GRP fra rå data (skal vises rundt om deres institut)
    for nid, m in raw_nodes.items():
        if m.get("type") == "grp":
            node_meta[nid] = m

    # 3) Edge-kilde: brug de RÅ grp-kanter (ikke de inst-inst aggregerede)
    edge_source = list(raw_edges)

# I = kun institutter
elif mode == "I":
    # 1) Filtrér rå GRP-noder efter kategori (uden sizes)
    raw_grp_for_I = {
    nid: m for nid, m in raw_nodes.items()
    if m.get("type") == "grp"
    }


    # 2) Merge KUN gruppenoder til institutter
    nm, es = merge_grp_to_inst(raw_grp_for_I, raw_edges)

    # 3) I-mode viser kun inst-noder fra MERGE
    node_meta = {
        nid: m for nid, m in nm.items()
        if m.get("type") == "inst"
    }

    # 4) Brug de merge-oprettede kanter
    edge_source = es





# G = kun grupper (ingen merges)
elif mode == "G":
    # summer grupper på tværs af hele KU
    merged = {}
    sizes  = {}
    for nid, m in raw_nodes.items():
        if m["type"] != "grp":
            continue
        grp = m["grp"]
        if grp not in merged:
            merged[grp] = {"type":"grp", "grp":grp, "size":0}
            sizes[grp]  = 0
        sizes[grp] += m["size"]

    # indsæt summeret størrelse
    for g,s in sizes.items():
        merged[g]["size"] = s

    node_meta = merged


    # --- SUM EDGES PÅ GRP-NIVEAU ---
    edge_dict = {}   # (grp_u, grp_v) -> weight

    for u_raw, v_raw, w in raw_edges:
        mu = raw_nodes[u_raw]
        mv = raw_nodes[v_raw]
        grp_u = mu["grp"]
        grp_v = mv["grp"]
        if grp_u == grp_v:
            continue
        key = tuple(sorted((grp_u, grp_v)))
        edge_dict[key] = edge_dict.get(key, 0) + w

    edge_source = [(u, v, w) for (u, v), w in edge_dict.items()]

def passes_category_filters(m):
    fac = m.get("fac")
    inst = m.get("inst")
    grp = m.get("grp")
    t = m.get("type")

    # Fakultet
    if show_fac and selected_facs and fac not in selected_facs:
        return False

    # Institut
    if show_inst and selected_insts and inst not in selected_insts:
        return False

    # GRP-filter må ALDRIG fjern GRP i FI/IG-mode
    if mode in ("FI","IG"):
        return True

    # Ellers GRP-filter normalt
    if t == "grp" and show_grp and selected_grps:
        if grp not in selected_grps:
            return False
    
    if mode == "I":
        return True

    return True


    # --- Case C: INST i andre modes (FG, FIG, I, F, G) skal IKKE filtreres af GRP
    # (Derfor ingen logik her)

    # --- FAC-noder filtreres aldrig på GRP
    return True






# Kandidatnoder efter KUN kategorifiltre
pre_size_nodes = {nid: m for nid, m in node_meta.items() if passes_category_filters(m)}

if not pre_size_nodes:
    st.error("Ingen noder matcher valgte filtre.")
    st.stop()


def size_relevant_in_mode(m, mode):
    t = m.get("type", "grp")

    if mode == "FI":
        # KUN inst-noder må size-filtreres i FI
        return t == "inst"

    if t == "fac":
        return mode == "F"

    if t == "inst":
        return mode in ("FI", "IG", "I")

    if t == "grp":
        return mode in ("G", "FG", "FIG") 

    return False


# pre_size_nodes eksisterer allerede: kun kategori-passede noder
candidate_items = [(nid, int(m.get("size", 0)))
                   for nid, m in pre_size_nodes.items()
                   if size_relevant_in_mode(m, mode)]

candidate_sizes = [size for _, size in candidate_items]

# --- 2) Dynamiske grænser (garantér min < max) ---
if candidate_sizes:
    vis_min = min(candidate_sizes)
    vis_max = max(candidate_sizes)
else:
    vis_min, vis_max = 0, 1

if vis_min == vis_max:
    # gør slider gyldig
    if vis_min == 0:
        vis_max = 1
    else:
        vis_min = vis_min - 1


author_ver = (mode, tuple(sorted(candidate_items)))

# --- 4) Reset slider, hvis versionen har ændret sig ---
if st.session_state.get("author_version") != author_ver:
    st.session_state["author_range"] = (vis_min, vis_max)
    st.session_state["author_version"] = author_ver

# --- 5) Opret slider NU (efter evt. reset) ---
#col_author, col_edges = st.columns([3, 3])

with cols[1]:
    author_min, author_max = st.slider(
        "Filter: antal forfatterbidrag (dynamisk)",
        min_value=vis_min,
        max_value=vis_max,
        value=st.session_state.get("author_range", (vis_min, vis_max)),
        key="author_minmax_slider"
    )
    # opdater state så næste rerun husker brugerens evt. manuelle valg
    st.session_state["author_range"] = (author_min, author_max)

with cols[1]:
    edge_min, edge_max = st.slider(
        "Filter: antal publikationer (kanter)",
        min_value=int(global_min_w),
        max_value=int(global_max_w),
        value=(int(global_min_w), int(global_max_w)),
        step=1,
        key="edge_slider"
    )
    edge_scale = st.slider(
        "Kantvægt",
        min_value=1.0,
        max_value=15.0,
        value=6.0,
        step=0.1,
        key="edge_scale_slider"
    )

# Brug derefter author_min/author_max i din størrelsesfiltrering
dyn_author_min, dyn_author_max = author_min, author_max




def node_passes_size(m):
    t = m.get("type")

    if t == "fac":
        return True

    if mode == "FI":
        # inst-noder MÅ IKKE size-filtreres i FI-mode
        if t == "inst":
            return True
        # grp skal aldrig size-filtreres i FI
        return True

    if mode == "IG":
        if t == "inst":
            return dyn_author_min <= m.get("size",0) <= dyn_author_max
    else:
        return True

    return True


node_meta = {nid: m for nid, m in pre_size_nodes.items() if node_passes_size(m)}

if not node_meta:
    st.error("Ingen noder matcher de valgte filtre (kategori + størrelse).")
    st.stop()



# Vælg noder
def node_passes_filters(nid, m):
    fac = m.get("fac")
    inst = m.get("inst")
    grp = m.get("grp")
    t   = m.get("type")

    if mode == "IG":
        return True

    if show_fac and selected_facs and fac not in selected_facs:
        return False

    if show_inst and selected_insts and inst not in selected_insts:
        return False

    # GRP node filtering
    if t == "grp" and show_grp and selected_grps:
        if grp not in selected_grps:
            return False

    # INST nodes must NOT be GRP-filtered in FI/IG
    if t == "inst" and show_grp and selected_grps and mode not in ("FI", "IG"):
        # inst-noder i fx FG/FIG kan godt sorteres via grp
        # men i FI/IG må de ikke filtreres på grp
        if grp and grp not in selected_grps:
            return False
    

    return True


node_meta = {nid: m for nid, m in node_meta.items() if node_passes_filters(nid, m)}

if not node_meta:
    st.error("Ingen noder matcher de valgte filtre.")
    st.stop()

def edge_passes_filters(u, v, w):
    if u not in node_meta or v not in node_meta:
        return False
    return True

edge_source = [
    (u, v, w) for (u, v, w) in edge_source if edge_passes_filters(u, v, w)
]
edge_source = [(u, v, w) for (u, v, w) in edge_source if edge_within_weight_range(w)]

if mode == "FIG":
    # inst-noder skal IKKE vises, kun bruges i layout
    nodes_keep = {nid for nid, m in node_meta.items() if m.get("type") != "inst"}
elif mode == "IG":
    nodes_keep = {nid for nid, m in node_meta.items() if m.get("type") == "grp"}
else:
    nodes_keep = set(node_meta.keys())

def edge_type(u, v):
    # I mode G findes fac ikke -> alle edges er "group"
    if mode == "G":
        return "group"

    # alle andre modes har fac
    fu = node_meta[u]["fac"]
    fv = node_meta[v]["fac"]
    return "intra" if fu == fv else "inter"


# Behold kanter, hvor "begge ender" er valgte
edges_keep = []

for u, v, w in edge_source:
    if u not in node_meta or v not in node_meta:
        continue
    if u not in nodes_keep or v not in nodes_keep:
        continue
    
    et = edge_type(u, v)

    if et == "intra" and not show_intra:
        continue
    if et == "inter" and not show_inter:
        continue

    if not (edge_min <= w <= edge_max):
        continue

    edges_keep.append((u, v, w))

nodes_keep = {nid for nid in nodes_keep if any(nid in (u, v) for (u, v, w) in edges_keep)}





pos = {}

R_fac = 800
R_inst = 350
R_grp = 150
R_g = 600 

finished_g_layout = False

# --- FACULTY LAYOUT ---
if mode in ("F", "FI", "FG", "FIG"):
    faculties = sorted({m["fac"] for m in node_meta.values() if "fac" in m})
    fac_centers = {}
    k = max(1, len(faculties))
    for i, fac in enumerate(faculties):
        theta = 2 * pi * i / k
        fac_centers[fac] = (R_fac * cos(theta), R_fac * sin(theta))
else:
    fac_centers = {}


# --- INSTITUTE LAYOUT ---
inst_centers = {}

if mode == "FI":
    insts = [(m["fac"], m["inst"]) for m in node_meta.values() if m.get("type") == "inst"]
    inst_by_fac = {}
    for fac, inst in insts:
        inst_by_fac.setdefault(fac, []).append(inst)
    for fac, insts in inst_by_fac.items():
        cx, cy = fac_centers.get(fac, (0, 0))
        k = max(1, len(insts))
        for j, inst in enumerate(insts):
            theta = 2 * pi * j / k
            inst_centers[(fac, inst)] = (cx + R_inst * cos(theta), cy + R_inst * sin(theta))

elif mode == "I":
    insts = sorted({(m["fac"], m["inst"]) for m in node_meta.values() if m.get("type") == "inst"})
    k = max(1, len(insts))
    for i, (fac, inst) in enumerate(insts):
        theta = 2 * pi * i / k
        inst_centers[(fac, inst)] = (R_fac * cos(theta), R_fac * sin(theta))

elif mode == "FIG":
    # Brug KUN de inst., der har GRP-noder tilbage efter filtre
    insts_by_fac = {}
    for nid, m in node_meta.items():
        if m.get("type") == "grp":
            insts_by_fac.setdefault(m["fac"], []).append(m["inst"])

    for fac, insts in insts_by_fac.items():
        cx, cy = fac_centers.get(fac, (0, 0))
        insts = sorted(set(insts))
        k = max(1, len(insts))
        for j, inst in enumerate(insts):
            theta = 2 * pi * j / k
            inst_centers[(fac, inst)] = (cx + R_inst * cos(theta), cy + R_inst * sin(theta))

elif mode == "IG":
    insts = sorted({(m["fac"], m["inst"]) for m in node_meta.values() if m.get("type") == "inst"})
    k = max(1, len(insts))
    for i, (fac, inst) in enumerate(insts):
        theta = 2 * pi * i / k
        inst_centers[(fac, inst)] = (R_fac * cos(theta), R_fac * sin(theta))

elif mode == "G":
    cluster = sorted(nodes_keep)
    k = max(1, len(cluster))

    for j, nid2 in enumerate(cluster):
            theta = 2 * pi * j / k
            pos[nid2] = (R_g * cos(theta), R_g * sin(theta))
    
    finished_g_layout = True


# --- NØGLEN: Læg kun noder ud for dem, der faktisk tegnes ---
for nid in nodes_keep:
    if finished_g_layout:
        break

    m = node_meta[nid]
    t = m.get("type")

    if t == "fac":
        # kræver fac_centers er sat i relevante modes
        pos[nid] = fac_centers.get(m["fac"], (0, 0))

    elif t == "inst":
        # i FIG-mode findes inst-noder ikke i nodes_keep, så vi ender ikke her
        center = inst_centers.get((m["fac"], m["inst"]))
        if center is None:
            # spring inst uden center over (kan ske ved skæve filtre)
            continue
        pos[nid] = center

    elif t == "grp":
        fac = m.get("fac", None)
        inst = m.get("inst", None)

        # center: i FIG-mode ligger GRP rundt om institut-centre
        if mode in ("FIG",):
            center = inst_centers.get((fac, inst))
            if center is None:
                continue
            cx, cy = center

            # alle GRP-noder i samme institut
            cluster = [
                nid2 for nid2 in nodes_keep
                if node_meta[nid2].get("type") == "grp"
                and node_meta[nid2].get("fac") == fac
                and node_meta[nid2].get("inst") == inst
            ]

        elif mode == "FG":
            cx, cy = fac_centers.get(fac, (0, 0))
            cluster = [
                nid2 for nid2 in nodes_keep
                if node_meta[nid2].get("type") == "grp"
                and node_meta[nid2].get("fac") == fac
            ]

        elif mode in ("FI", "IG", "I"):
            center = inst_centers.get((fac, inst))
            if center is None:
                continue
            cx, cy = center
            cluster = [
                nid2 for nid2 in nodes_keep
                if node_meta[nid2].get("type") == "grp"
                and node_meta[nid2].get("fac") == fac
                and node_meta[nid2].get("inst") == inst
            ]

        else:
            cx, cy = (0, 0)
            cluster = [nid]

        # --- LAV CIRKEL ---
        cluster = sorted(cluster)
        k = max(1, len(cluster))
        j = cluster.index(nid)
        theta = 2 * pi * j / k
        pos[nid] = (cx + R_grp * cos(theta), cy + R_grp * sin(theta))




# Skaler node-størrelser (til pixels)
raw_sizes = [node_meta[n].get("size", 1) for n in nodes_keep] or [1]
s_min, s_max = min(raw_sizes), max(raw_sizes)

def scale_size(val, vmin = s_min, vmax = s_max, px_min = global_min_auth, px_max = global_max_auth):
    if vmax <= vmin:
        return (px_min + px_max) / 2
    t = (val - vmin) / (vmax - vmin)
    return px_min + t*(px_max - px_min)


# Opret PyVis-netværket
net = Network(height = "700px", width = "100%", directed = False)
net.toggle_physics(False)






for nid in nodes_keep:
    meta = node_meta[nid]
    fac = meta.get("fac", "")
    inst = meta.get("inst", "")
    grp = meta.get("grp", "")
    size = meta.get("size", "NA")

    parts = []

# kun tilføj hvis de er relevante / ikke skjult / ikke tomme

    if mode in ("F", "FI", "FG", "FIG", "IG", "I"):     # modes hvor fac er relevant
        if fac:
            parts.append(fac)

    if mode in ("FI", "FIG", "IG", "I"):               # modes hvor inst er relevant
        if inst:
            parts.append(inst)

    if mode in ("G", "FG", "FIG", "IG"):               # modes hvor grp er relevant
        if grp:
            parts.append(grp)

    # altid tilføj size
    #parts.append(f"{size}")

    # lav tooltip string
    title_text = " | ".join(parts) + f"\n Forfatterbidrag: {size}"

    # ---------- MODE G: ingen fakultetsfarver, ingen fac/inst ----------
    if mode == "G":
        color = "#888888"           # neutral grå
        label = grp
    elif mode in ("I", "FI"):
        label = inst
        insts = sorted({m["inst"] for m in node_meta.values() if m["fac"] == fac})
        rank = insts.index(inst)
        k = len(insts)
        factor = 0.9 + 0.3 * (rank / max(1, k - 1))
        base = faculty_base_colors.get(fac, "black")
        color = adjust_color(base, factor)
    elif mode == "F":    
        label = fac
        base = faculty_base_colors.get(fac, "black")
        color = base  # faculty nodes = pure faculty color
    else:
        fac = meta["fac"]           # ← dette findes KUN i ikke-G-mode
        base = faculty_base_colors.get(fac, "black")
        lvl = hierarki.get(grp, lvl_min)
        lightness_factor = 1.2 + 1.2 * (1 - (lvl / max(hierarki.values())))
        saturation_factor = 0.6 + 0.4 * (1 - (lvl / max(hierarki.values())))
        color = adjust_color(base, lightness_factor, saturation_factor)
        label = grp

    x, y = pos[nid]
    size_px = scale_size_log(meta.get("size", 1))

    net.add_node(
        nid,
        label = label,
        x = x, y = y,
        size = size_px,
        color = color,
        title = title_text,
        physics = False,
        font = "30px"
    )


fac_sizes = {}
inst_sizes = {}
grp_sizes = {}

# Saml sizes pr. niveau
for nid in nodes_keep:
    m = node_meta[nid]
    size = m.get("size", 0)

    fac = m.get("fac")
    inst = m.get("inst")
    grp = m.get("grp")
    t   = m.get("type")

    # Stillingsgrupper er dem med type="grp"
    if t == "grp":
        grp_sizes.setdefault(grp, []).append(size)

    # Institutter
    if inst:
        inst_sizes.setdefault(inst, []).append(size)

    # Fakulteter
    if fac:
        fac_sizes.setdefault(fac, []).append(size)

# Beregn gennemsnit
fac_avg_size  = {fac: sum(vals)/len(vals) for fac, vals in fac_sizes.items()}
fac_tot_size = {fac: sum(vals) for fac, vals in fac_sizes.items()}
inst_avg_size = {inst: sum(vals)/len(vals) for inst, vals in inst_sizes.items()}
inst_tot_size = {inst: sum(vals) for inst, vals in inst_sizes.items()}
grp_avg_size  = {grp: sum(vals)/len(vals) for grp, vals in grp_sizes.items()}
grp_tot_size = {grp: sum(vals) for grp, vals in grp_sizes.items()}


G = nx.Graph()

# Tilføj noder
for nid in nodes_keep:
    G.add_node(nid)

# Tilføj kanter (med weight)
for u, v, w in edges_keep:
    G.add_edge(u, v, weight=w)




def aggregate_centrality_by(meta_key, nodes_list, node_meta, weighted_deg, bet_cent):
    """
    Aggregerer centralitet (weighted degree + betweenness) pr. værdi i node_meta[meta_key],
    baseret på de noder, der faktisk findes i nodes_list.
    Returnerer to sorterede lister: (key, wd_sum) og (key, bc_sum).
    """
    acc_wd = {}
    acc_bc = {}

    for n in nodes_list:
        meta = node_meta.get(n, {})
        key = meta.get(meta_key, "")
        if not key:
            continue
        acc_wd[key] = acc_wd.get(key, 0.0) + float(weighted_deg.get(n, 0.0))
        acc_bc[key] = acc_bc.get(key, 0.0) + float(bet_cent.get(n, 0.0))

    wd_sorted = sorted(acc_wd.items(), key=lambda x: -x[1])
    bc_sorted = sorted(acc_bc.items(), key=lambda x: -x[1])
    return wd_sorted, bc_sorted


grp_nodes = [n for n, m in node_meta.items() if m.get("type") == "grp"]
inst_nodes = [n for n, m in node_meta.items() if m.get("type") == "inst"]
fac_nodes  = [n for n, m in node_meta.items() if m.get("type") == "fac"]


weighted_deg = dict(G.degree(weight="weight"))
bet_cent = nx.betweenness_centrality(G, weight="weight", normalized=True)

if fac_nodes:
    source_nodes_for_fac = fac_nodes
elif inst_nodes:
    source_nodes_for_fac = inst_nodes
else:
    source_nodes_for_fac = grp_nodes  # fallback

faculty_wd_sorted, faculty_bs_sorted = aggregate_centrality_by(
    meta_key="fac",
    nodes_list=source_nodes_for_fac,
    node_meta=node_meta,
    weighted_deg=weighted_deg,
    bet_cent=bet_cent
)

inst_wd_sorted, inst_bs_sorted = aggregate_centrality_by(
    meta_key="inst",
    nodes_list=(inst_nodes or grp_nodes),  # brug inst hvis de findes, ellers grp
    node_meta=node_meta,
    weighted_deg=weighted_deg,
    bet_cent=bet_cent
)

### hvad sker der her?!
if mode in ("IG", "FIG"):
    # inst-centr. skal opsummeres fra GRP-noder (inst-noder er ikke i G)
    nodes_for_inst_centrality = grp_nodes
else:
    nodes_for_inst_centrality = (inst_nodes or grp_nodes)

inst_wd_sorted, inst_bs_sorted = aggregate_centrality_by(
    meta_key="inst",
    nodes_list=nodes_for_inst_centrality,
    node_meta=node_meta,
    weighted_deg=weighted_deg,
    bet_cent=bet_cent
)


grp_wd_sorted = sorted([(n, float(weighted_deg.get(n, 0.0))) for n in grp_nodes],
                       key=lambda x: -x[1])
grp_bs_sorted = sorted([(n, float(bet_cent.get(n, 0.0))) for n in grp_nodes],
                       key=lambda x: -x[1])



#grp_wd_sorted = sorted([(n, weighted_deg.get(n,0)) for n in grp_nodes], key=lambda x:-x[1])
#grp_bs_sorted = sorted([(n, bet_cent.get(n,0)) for n in grp_nodes],  key=lambda x:-x[1])

if edges_keep:
    max_w_cur = max(w for _, _, w in edges_keep)
else:
    max_w_cur = 1.0

for u, v, w in edges_keep:
    width = 6 * edge_scale * (w / max_w_cur)

    if mode == "G":
        et = "group"
        col = "gray"

    else:
        fu = node_meta[u]["fac"]
        fv = node_meta[v]["fac"]
        et = "intra" if fu == fv else "inter"
        col = "black" if et == "inter" else adjust_color(faculty_base_colors[fu], 0.25)

    net.add_edge(
        u, v,
        width = width,
        color = add_alpha(col, 0.4),
        title = f"Publikationer: {int(w)} ({et})"
    )

fac_strength = {}
fac_edge_acc = {}

for u, v, w in edges_keep:
    fu = node_meta[u].get("fac")
    fv = node_meta[v].get("fac")

    if not fu or not fv:
        continue

    tu = node_meta[u].get("type")
    tv = node_meta[v].get("type")

    if mode == "F":
        if tu == "fac" and tv == "fac":
            fac_strength[fu] = fac_strength.get(fu, 0.0) + w
            fac_strength[fv] = fac_strength.get(fv, 0.0) + w

            key = tuple(sorted((fu, fv)))
            fac_edge_acc[key] = fac_edge_acc.get(key, 0.0) + w
    
    elif mode == "FI":       
        if (tu == "fac" and tv == "fac") or (tu == "inst" and tv == "inst"):
            fac_strength[fu] = fac_strength.get(fu, 0.0) + w
            fac_strength[fv] = fac_strength.get(fv, 0.0) + w

            key = tuple(sorted((fu, fv)))
            fac_edge_acc[key] = fac_edge_acc.get(key, 0.0) + w

    
    elif mode == "FG":
        if tu == "grp" and tv == "grp":
            fac_strength[fu] = fac_strength.get(fu, 0.0) + w
            fac_strength[fv] = fac_strength.get(fv, 0.0) + w

            key = tuple(sorted((fu, fv)))
            fac_edge_acc[key] = fac_edge_acc.get(key, 0.0) + w


    elif mode == "FIG":       
        fac_strength[fu] = fac_strength.get(fu, 0.0) + w
        fac_strength[fv] = fac_strength.get(fv, 0.0) + w

        key = tuple(sorted((fu, fv)))
        fac_edge_acc[key] = fac_edge_acc.get(key, 0.0) + w

 
    else:
        fac_strength = {}
        fac_edge_acc = {}
        break

faculty_wd_sorted = sorted(fac_strength.items(), key=lambda x: -x[1])
 

for (fu, fv), w_sum in fac_edge_acc.items():
    G.add_edge(fu, fv, weight=w_sum)

# Tilføj evt. isolerede fakulteter (f.eks. hvis kun én node findes)
for n, m in node_meta.items():
    if m.get("type") in ("fac", "inst", "grp"):
        fac = m.get("fac")
        if fac:
            G.add_node(fac)

# 3) Beregn fakultets-betweenness
if len(G.nodes) >= 1:
    fac_bet = nx.betweenness_centrality(G, weight="weight", normalized=True)
    fac_bs_sorted = sorted(fac_bet.items(), key=lambda x: -x[1])
else:
    fac_bs_sorted = []


# Byg graf
temp_path = os.path.join(tempfile.gettempdir(), f"pyvis_{year}.html")
net.write_html(temp_path)
with open(temp_path, "r", encoding="utf-8") as f:
    html = f.read()

st.components.v1.html(html, height=800, scrolling=True)


total_pubs = sum(w for _, _, w in edges_keep)
intra_pubs = sum(w for u, v, w in edges_keep if edge_type(u, v) == "intra")
inter_pubs = sum(w for u, v, w in edges_keep if edge_type(u, v) == "inter")




tabs_by_mode = {
    "F": ["Oversigt", "Fakulteter", "Centralitet", "Netværksstruktur", "Datagrundlag"],
    "FI": ["Oversigt", "Fakulteter", "Institutter", "Centralitet", "Netværksstruktur", "Datagrundlag"],
    "FIG": ["Oversigt", "Fakulteter", "Institutter", "Stillingsgrupper", "Centralitet", "Netværksstruktur", "Datagrundlag"],
    "FG": ["Oversigt", "Fakulteter", "Stillingsgrupper", "Centralitet", "Netværksstruktur", "Datagrundlag"],
    "IG": ["Oversigt", "Institutter", "Stillingsgrupper", "Centralitet", "Netværksstruktur", "Datagrundlag"],
    "I": ["Oversigt", "Institutter", "Centralitet", "Netværksstruktur", "Datagrundlag"],
    "G": ["Oversigt", "Stillingsgrupper", "Centralitet", "Netværksstruktur", "Datagrundlag"]
}

tabs_to_show = tabs_by_mode.get(mode, ["Basisstatistik"])

tabs = st.tabs(tabs_to_show)

tabs_dict = {name: tab for name, tab in zip(tabs_to_show, tabs)}

with tabs_dict["Oversigt"]:
    st.subheader("Statistik for udsnittet")
    st.markdown(
f"""
Dette afsnit giver et samlet overblik over publikationsmængden og de overordnede 
sampubliceringsmønstre i det valgte udsnit for **{year}**. Tallene afspejler alene summen af
publikationer mellem de noder, der er inkluderede på baggrund af de valgte filtre
(fakulteter, institutter og stillingsgrupper).

**Intra-fakultetspublikationer** dækker over samarbejde inden for samme fakultet, mens 
**Inter-fakultetspublikationer** viser samarbejde på tværs af fakulteter. Sammen med
det samlede antal publikationer giver disse indikatorer en indledende vurdering af
både udsnittets interne sammenhængskraft og graden af samarbejde på tværs af KU's 
faglige miljøer. 
""")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total antal publikationer(kanter)", 
        int(total_pubs),
        help = "Summen af publikationer mellem alle grupper i det filtrerede netværk"
        )
    
    with col2:
        st.metric("Intra-fakultetspublikationer",
        int(intra_pubs),
        help = "Publikationer, hvor begge forfattergrupper er fra forskellige samme fakultet"
        )
    
    with col3:
        st.metric("Inter-fakultetspublikationer",
        int(inter_pubs),
        help = "Publikationer, hvor begge forfattergrupper er fra forskellige fakulteter"  
        )

    st.subheader("**Top sampubliceringer i udsnittet**")
    st.markdown(
f"""
Nedenfor er der en oversigt over grupper med flest sampubliceringer. 
""")

    max_n = len(edges_keep)
    default_n = min(10, max_n)
    
    col_top, col_dummy = st.columns([1, 5])
    top_n = st.number_input(
        "**Hvor mange sampubliceringer skal vises?**",
        min_value = 1,
        max_value = max_n,
        value = default_n,
        step = 1,
        key = f"top_edges_n_{year}_{mode}"
    )

    #st.markdown(f"**Top-{top_n} sampubliceringer i udsnittet**")

    if not edges_keep:
        st.info("Ingen sampubliceringer matcher det valgte udsnit.")
    else:
        top_edges = sorted(edges_keep, key = lambda x: -x[2])[:top_n]

        top_edge_rows = []
        for u, v, w in top_edges:
            mu = node_meta.get(u, {})
            mv = node_meta.get(v, {})

            def make_label(m):
                parts = []
                if m.get("fac"):
                    parts.append(m["fac"])
                if m.get("inst"):
                    parts.append(m["inst"])
                if m.get("grp"):
                    parts.append(m["grp"])
                return " | ".join(parts)
            
            label_u = make_label(mu)
            label_v = make_label(mv)

            et_type = "Intra-fakultet" if mu.get("fac") == mv.get("fac") else "Inter-fakultet"

            top_edge_rows.append({
                "Den ene node": label_u,
                "Den anden node": label_v,
                "Antal publikationer": int(w),
                "Type": et_type,
                "mode": mode
            })

        top_edge_schema = [
            ("Den ene node", pa.string()),
            ("Den anden node", pa.string()),
            ("Antal publikationer", pa.int64()),
            ("Type", pa.string()),
            ("mode", pa.string())
        ]

        top_edge_table = build_table(top_edge_rows, top_edge_schema)

        st.dataframe(top_edge_table, width = "stretch", hide_index = True)

        csv_bytes = rows_to_csv_bytes(top_edge_rows, [name for name, _ in top_edge_schema])
        st.download_button(
            "Download (.csv)",
            data = csv_bytes,
            file_name = f"Top_{top_n}_edges_{year}_{mode}.csv")










if "Fakulteter" in tabs_dict:
    with tabs_dict["Fakulteter"]:
        if mode in ("FIG", "FI", "FG", "F"):
            st.subheader("Fakulteters forfatterbidrag")

            st.markdown(
f"""
Den her sektion giver et samlet overblik over **fakulteternes forfatterbidrag** i det 
valgte udsnit for **{year}**. Resultaterne afspejler de filtre, der er valgt i sidepanelet 
(fakulteter, institutter og stillingsgrupper) og viser derfor kun aktivitet inden for
det aktuelle udsnit.
""")

            st.markdown(
f"""**Fakulteters samlede forfatterbidrag i udsnittet** angiver den samlede volumen af
publikationer, som de enkelte fakulteter indgår i inden for udsnittet. Disse tal påvirkes 
både af den samlede publikationsvolumen og af sammensætningen af de viste institutter og
stillingsgrupper. 
""")
            cols = st.columns(4)
            i = 0
            for fac, tot in fac_tot_size.items():
                with cols[i % 4]:
                    st.metric(
                        fac,
                        f"{int(tot)}",
                        help = f"Samlede antal publikationer i {year}"
                    )
                i += 1

            fac_total_rows = []
            for fac in fac_order:
                if fac in fac_tot_size:
                    fac_total_rows.append({
                        "Fakultet": fac,
                        "Samlet forfatterbidrag": int(fac_tot_size[fac]),
                        "mode": mode,
                    })
            
            fac_total_schema = [
                ("Fakultet", pa.string()),
                ("Samlet forfatterbidrag", pa.int64()),
                ("mode", pa.string())
            ]
            
            fac_total_table = build_table(fac_total_rows, fac_total_schema)
            with st.expander("Se tabel over fakulteternes forfatterbidrag"):
                st.dataframe(fac_total_table, width = "stretch", hide_index=True)

                csv_bytes = rows_to_csv_bytes(fac_total_rows, [name for name, _ in fac_total_schema])
                st.download_button(
                    "Download (.csv)", 
                    data = csv_bytes,
                    file_name = f"forfatterbidrag_fakulteter_total_{year}_{mode}.csv"
                )
            
            st.markdown(
f"""**Fakulteters gennemsnitlige forfatterbidrag i udsnittet** normaliserer publikationsmængden
i forhold til antal unikke forfattere i udsnittet. 
""")
            cols = st.columns(4)
            i = 0
            for fac, avg in fac_avg_size.items():
                with cols[i % 4]:
                    st.metric(
                        fac,
                        f"{int(avg)}",
                        help = f"Gennemsnitlige antal publikationer i {year}"
                    )
                i += 1

            fac_avg_rows = []
            for fac in fac_order:
                if fac in fac_avg_size:
                    fac_avg_rows.append({
                        "Fakultet": fac,
                        "Gennemsnitlige forfatterbidrag": (fac_avg_size[fac]),
                        "mode": mode,
                    })
            
            fac_avg_schema = [
                ("Fakultet", pa.string()),
                ("Gennemsnitlige forfatterbidrag", pa.int64()),
                ("mode", pa.string())
            ]
            
            fac_avg_table = build_table(fac_avg_rows, fac_avg_schema)
            with st.expander("Se tabel over fakulteternes gennemsnitlige forfatterbidrag"):
                st.dataframe(fac_avg_table, width = "stretch", hide_index=True)

                csv_bytes = rows_to_csv_bytes(fac_avg_rows, [name for name, _ in fac_avg_schema])
                st.download_button(
                    "Download (.csv)", 
                    data = csv_bytes,
                    file_name = f"forfatterbidrag_fakulteter_gns_{year}_{mode}.csv"
                )
        

if "Institutter" in tabs_dict:
    with tabs_dict["Institutter"]:
        st.subheader("Institutternes forfatterbidrag")

        st.markdown(
f""" 
Her er et samlet overblik over **institutternes forfatterbidrag** i det valgte
udsnit for **{year}**. Resultaterne afspejler de filtre, der er valgt i sidepanelet
(fakulteter, institutter og stillingsgrupper) og viser derfor kun aktivitet inden for
det aktuelle udsnit. 

**Det samlede forfatterbidrag** viser, hvor ofte institutterne indgår i publikationerne,
der indgår i netværket. Dette mål påvirkes både af institutstørrelse og af de aktuelle
fakultets- og stillingsgruppefiltre. XXX tjek faktiske tal  
""")
 
        inst_total_rows = []
        for inst in sorted(inst_tot_size.keys()):
            fac = institut_fakultets_map.get(inst, "")

            inst_total_rows.append({
                "Fakultet": fac,
                "Institut": inst,
                "Samlet forfatterbidrag": int(inst_tot_size[inst]),
                "mode": mode,
            })
            
        inst_total_schema = [
            ("Fakultet", pa.string()),
            ("Institut", pa.string()),
            ("Samlet forfatterbidrag", pa.int64()),
            ("mode", pa.string())
        ]
            
        inst_total_table = build_table(inst_total_rows, inst_total_schema)
        with st.expander("Se tabel over institutternes forfatterbidrag"):
            st.dataframe(inst_total_table, width = "stretch", hide_index=True)

            csv_bytes = rows_to_csv_bytes(inst_total_rows, [name for name, _ in inst_total_schema])
            st.download_button(
                "Download (.csv)", 
                data = csv_bytes,
                file_name = f"forfatterbidrag_institutter_total_{year}_{mode}.csv"
                )


        st.markdown(
f"""**Institutters gennemsnitlige forfatterbidrag** er publikationsvolumen for hvert valgte
institut normeret med antal unikke forfatterbidrag (i valgte stillingsgrupper) i pågældende institut. 
""")
        inst_avg_rows = []
        for inst in sorted(inst_avg_size.keys()):
            fac = institut_fakultets_map.get(inst, "")

            inst_avg_rows.append({
                "Fakultet": fac,
                "Institut": inst,
                "Gennemsnitligt forfatterbidrag": int(inst_avg_size[inst]),
                "mode": mode
            })

            inst_avg_schema = [
                ("Fakultet", pa.string()),
                ("Institut", pa.string()),
                ("Gennemsnitligt forfatterbidrag", pa.int64()),
                ("mode", pa.string())
            ]
        
        inst_avg_table = build_table(inst_avg_rows, inst_avg_schema)
        with st.expander("Se tabel over institutternes gennemsnitlige forfatterbidrag"):
            st.dataframe(inst_avg_table, width = "stretch", hide_index = True)

            csv_bytes = rows_to_csv_bytes(inst_avg_rows, [name for name, _ in inst_avg_schema])
            st.download_button(
                "Download (.csv)",
                data = csv_bytes,
                file_name = f"forfatterbidrag_institutter_gns_{year}_{mode}.csv"
            )


def author_passes_filters(fac_list, inst_list, grp, selected_facs, selected_insts, selected_grps):
    # Fakultet
    if selected_facs:
        if not any(f in selected_facs for f in fac_list):
            return False

    # Institut
    if selected_insts:
        if not any(i in selected_insts for i in inst_list):
            return False

    # Stillingsgruppe
    if selected_grps:
        if grp not in selected_grps:
            return False

    return True


if "Stillingsgrupper" in tabs_dict:
    with tabs_dict["Stillingsgrupper"]:
        st.subheader("Stillinggruppernes forfatterbidrag")
        min_tot_grp = min(grp_tot_size, key = grp_tot_size.get)
        min_tot_val = grp_tot_size[min_tot_grp]

        max_tot_grp = max(grp_tot_size, key = grp_tot_size.get)
        max_tot_val = grp_tot_size[max_tot_grp]
        #abs_one = counts[max_fac][year]["1"]

        st.markdown(f"""
Her i afsnittet gives et samlet overblik over stillingsgruppernes bidrag i
sampublceringsnetværket for **{year}**. Fokus er både på det samlede og 
gennemsnitlige forfatterbidrag samt på stillinggruppernes placering i 
forfatterrækkefølgerne. 

**Stillinggruppernes samlede forfatterbidrag** viser, hvor ofte de enkelte 
stillingsgrupper indgår i publikationerne i udsnittet. På tværs af årene ses det,
at {max_tot_grp}-stillingsgruppen publicerer den største volumen, mens 
{min_tot_grp}-stillingsgruppen XXX tjek faktiske tal.
    """)

        cols = st.columns(4)
        i = 0
        for grp, tot in grp_tot_size.items():
            with cols[i % 4]:
                st.metric(
                    grp,
                    f"{int(tot)}",
                    help = f"Antal publikationer i {year} for denne stillingsgruppe"
                )
            i += 1

        st.markdown(f"""
**Stillinggruppernes gennemsnitlige forfatterbidrag** i udsnittet er normeret i 
forhold til antal unikke forfattere i pågældende år.  
""")
        cols = st.columns(4)
        i = 0
        for grp, avg in grp_avg_size.items():
            with cols[i % 4]:
                st.metric(
                    grp,
                    f"{int(avg)}",
                    help = f"Gennemsnitligt antal publikationer i {year} for denne stillingsgruppe"
                )
            i += 1


        st.subheader("Stillinggruppers placering i forfatterrækkefølger")
        st.markdown(f""" Nedenfor vises stillinggruppernes placering i publikationernes
forfatterrækkefølger i **{year}**. Kun publikationer med mindst to KU-forfattere indgår i analysen,
da disse pr. definition repræsenterer sampublicering. 

Hvis der derudover er filtreret på fakulteter, institutter eller stillingsgrupper, 
medregnes en publikation kun, hvis **mindst én KU-forfatter** på publikationen falder
inden for det valgte udsnit. I givet fald indgår hele publikationens forfatterrækkefølge
i opgørelsen.

På den baggrund viser figuren, hvor ofte hver stillingsgruppe indgår som hhv. **førsteforfatter**,
**mellemforfatter** og **sidsteforfatter**. 

XXX konkretiser med de faktiske tal - er der forskel i løbet af årene?
""")

        hierarki = {
            "Særlig stilling": -1,
            "Øvrige VIP (DVIP)": 0,
            "Ph.d.": 1,
            "Stillinger u. adjunktniveau": 2,
            "Postdoc": 3,
            "Adjunkt": 4,
            "Lektor": 5,
            "Professor": 6
        }
        
        stilling_map = {}
        with open("H:/Sampubliceringsanalyse/KU_stillingstyper.csv", encoding="utf-8") as f_stil:
            stil_reader = csv.DictReader(f_stil, delimiter=";")
            for row in stil_reader:
                raw = row["Stillingstype"].strip()
                med = row["medtages?"].strip()
                grp = row["Stillingsgruppe"].strip()
                if med == "1" and grp:
                    stilling_map[raw] = grp
        
        pos_counts = {
            "first": defaultdict(int),
            "middle": defaultdict(int),
            "last": defaultdict(int),
            "total": defaultdict(int)
        }

        csv_fil = "H:/Sampubliceringsanalyse/Data/KU_titles_CURIS_VIP_data.csv"

        with open(csv_fil, encoding = "utf-8") as f:
            reader = csv.DictReader(f, delimiter = ";")
            for row in reader:

                # Årfilter
                pub_year = int(row["Udgivelsesår"])
                if pub_year != year:
                    continue

                stil_raw = (row.get("Stillingsbetegnelse(r)") or "").strip()
                try:
                    stil_col = ast.literal_eval(stil_raw) if stil_raw else []
                except:
                    continue

                if len(stil_col) < 2:
                    continue
                
                fac_raw = (row.get("Fakultet(er)") or "").strip()
                inst_raw = (row.get("Institut(ter)") or "").strip()

                try:
                    fac_col = ast.literal_eval(fac_raw) if fac_raw else []
                except:
                    fac_col = []
                
                try:
                    inst_col = ast.literal_eval(inst_raw) if inst_raw else []
                except:
                    inst_col = []
                
                grp_list = []
                publication_has_matching_author = False

                for i, forfatter_stillinger in enumerate(stil_col):
                    highest_level = -1
                    highest_group = None

                    for raw_stil in forfatter_stillinger:
                        for item in raw_stil.split(","):
                            item = item.strip()
                            if not item or item in ["0", "-"]:
                                continue

                            if item in stilling_map:
                                grp = stilling_map[item]
                                lvl = hierarki.get(grp, -1)
                                if lvl > highest_level:
                                    highest_level = lvl
                                    highest_group = grp
                    
                    if not highest_group:
                        continue

                    fac_list = fac_col[i] if (i < len(fac_col)) else []
                    inst_list = inst_col[i] if (i < len(inst_col)) else []

                    fac_clean = [x.strip() for x in fac_list if x and x != "0"]
                    inst_clean = [x.strip() for x in inst_list if x and x != "0"]

                    if author_passes_filters(
                        fac_list = fac_clean,
                        inst_list = inst_clean,
                        grp = highest_group,
                        selected_facs = selected_facs,
                        selected_insts = selected_insts,
                        selected_grps = selected_grps
                        ):
                        publication_has_matching_author = True
                    
                    grp_list.append(highest_group)
                
                if not publication_has_matching_author:
                    continue

                for g in grp_list:
                    pos_counts["total"][g] += 1
                
                pos_counts["first"][grp_list[0]] += 1
                pos_counts["last"][grp_list[-1]] += 1

                for g in grp_list[1:-1]:
                    pos_counts["middle"][g] += 1
         
        rows = []
        for grp in all_groups:
            rows.append({
                "Stillingsgruppe": grp,
                "Førsteforfatter": pos_counts["first"][grp],
                "Mellemforfatter": pos_counts["middle"][grp],
                "Sidsteforfatter": pos_counts["last"][grp],
                "Total deltagelse": pos_counts["total"][grp]
            })

        schema = [
            ("Stillingsgruppe", pa.string()),
            ("Førsteforfatter", pa.int64()),
            ("Mellemforfatter", pa.int64()),
            ("Sidsteforfatter", pa.int64()),
            ("Total deltagelse", pa.int64())
        ]

        stillinger_sorted = sorted(all_groups, key = lambda g: hierarki.get(g, 999))

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name = "Førsteforfatter",
            y = stillinger_sorted,
            x = [pos_counts["first"][g] for g in stillinger_sorted],
            orientation = "h"
        ))

        fig.add_trace(go.Bar(
            name = "Mellemforfatter",
            y = stillinger_sorted,
            x = [pos_counts["middle"][g] for g in stillinger_sorted],
            orientation = "h"
        ))

        
        fig.add_trace(go.Bar(
            name = "Sidsteforfatter",
            y = stillinger_sorted,
            x = [pos_counts["last"][g] for g in stillinger_sorted],
            orientation = "h"
        ))


        fig.update_layout(
            barmode="stack",
            xaxis_title="Antal publikationer",
            yaxis_title="Stillingsgruppe",
            height=450,
            legend_title="Forfatterposition"
        )

        st.plotly_chart(fig, width = "stretch")

        with st.expander("Se tabel over stillinggruppernes forfatterpositioner"):
            table = build_table(rows, schema)
            st.dataframe(table, width = "stretch", hide_index = True)
            #csv_bytes = rows_to_csv_bytes(table, [name for name, _ in schema])
            #st.download_button("Download (.csv)", data = csv_bytes, file_name = f"klynger_{mode}_{year}.csv")
            # xx OPDATER RÆKKER I TABEL

def build_grp_table_by_mode(weighted_deg, bet_cent, node_meta, mode, year=None):
    import pyarrow as pa

    # maps: grp -> centralitet
    wd_map = {grp: float(val) for grp, val in weighted_deg} if weighted_deg else {}
    bc_map = {grp: float(val) for grp, val in bet_cent} if bet_cent else {}

    rows = []

    # altid: én række per GRP-node
    for nid, meta in node_meta.items():
        if meta.get("type") != "grp":
            continue

        grp = meta.get("grp", "")
        fac = meta.get("fac", "")
        inst = meta.get("inst", "")

        wd = wd_map.get(grp, 0.0)
        bc = bc_map.get(grp, 0.0)

        # FIG → grp + fac + inst
        if mode == "FIG":
            row = {"Stillingsgruppe": grp, "Fakultet": fac, "Institut": inst, "Weighted degree": wd, "betweenness": bc, "mode": mode}

            base_fields = ["Stillingsgruppe", "Fakultet", "Institut", "Weighted degree", "Betweenness", "mode"]

        # IG → grp + inst  (FAC SKAL IKKE MED)
        elif mode == "IG":
            row = {"Stillingsgruppe": grp, "Institut": inst, "Weighted degree": wd, "Betweenness": bc, "mode": mode}

            base_fields = ["Stillingsgruppe", "Institut", "Weighted degree", "Betweenness", mode]

        # FG → grp + fac (INST SKAL IKKE MED)
        elif mode == "FG":
            row = {"Stillingsgruppe": grp, "Fakultet": fac, "Weighted degree": wd, "Betweenness": bc, "mode": mode}

            base_fields = ["Stillingsgruppe", "Fakultet", "Weighted degree", "Betweenness", "mode"]

        # G → kun grp
        elif mode == "G":
            row = {"Stillingsgruppe": grp, "Weighted degree": wd, "Betweenness": bc, "mode": mode}

            base_fields = ["Stillingsgruppe", "Weighted degree", "Betweenness", "mode"]

        else:
            continue

        rows.append(row)

    # schema
    type_map = {
        "Stillingsgruppe": pa.string(),
        "Fakultet": pa.string(),
        "Institut": pa.string(),
        "Weighted degree": pa.float64(),
        "Betweenness": pa.float64(),
        "mode": pa.string()
    }
    schema = [(key, type_map[key]) for key in base_fields]

    return rows, schema



### Modularitet

if mode in ("F", "FI", "FG", "FIG"):
    comm_key = "fac"
elif mode in ("I", "IG"):
    comm_key = "inst"
else:
    comm_key = "grp"

communities = []
seen_groups = set()

for nid in nodes_keep:
    group_label = node_meta[nid].get(comm_key, "")
    if not group_label:
        continue
    if group_label not in seen_groups:
        # saml alle noder i samme gruppe
        members = [n for n in nodes_keep if node_meta[n].get(comm_key) == group_label]
        if members:
            communities.append(members)
            seen_groups.add(group_label)

abbrs = {
    "fac": "fakulteter",
    "inst": "institutter",
    "grp": "stillingsgrupper"
}

G2 = nx.Graph()
for u in nodes_keep:
    G2.add_node(u)
for u, v, w in edges_keep:
    G2.add_edge(u, v, weight = w)

density = nx.density(G2)

# Prædefineret modularitet
if communities and G2.number_of_edges() > 0:
    modularity_pre = modq(G2, communities, weight = "weight")
else:
    modularity_pre = float("nan")

# Greedy modularitet
if G2.number_of_edges() > 0 and G2.number_of_nodes() > 1:
    greedy_comms = list(greedy_modularity_communities(G2, weight = "weight"))
    modularity_greedy = modq(G2, greedy_comms, weight = "weight")
    n_comms = len(greedy_comms)
else:
    greedy_comms = []
    modularity_greedy = float("nan")
    n_comms = 0

if "Netværksstruktur" in tabs_dict: 
    with tabs_dict["Netværksstruktur"]:
        st.subheader("Modularitet og netværkstæthed for udsnittet")

        st.markdown(
""" 
**Modularitet** beskriver, i hvilken grad et netværk er opdelt i adskilte klynger
sammenlignet med, hvad man vil forvente tilfældigt. 
- Et netværk med høj modularitet (typisk over 0,3) er karakteriseret ved tydeligt 
adskilte grupper (klynger)
- Lav modularitet indikerer et mere sammenhængende og integreret netværk. 

I det valgte udsnit er modulariteten beregnet ud fra det mest aggregerede niveau,
der indgår i visningen (fakulteter, institutter eller stillingsgrupper 
afhængigt af valg i sidepanel). Det giver et mål for, hvor klart netværkets struktur
afspejler de overordnede organisatoriske enheder. 

Derudover beregnes en alternativ modularitet på baggrund af en data-drevet 
klyngealgoritme (*greedy modularitet*), som identificerer de klynger, der bedst
forklarer netværkets struktur uden foruddefinerede grupper. Det giver et supplerende
perspektiv på, om samarbejdsmønstrene følger de organisatoriske skel eller opstår 
på tværs.
""")

        st.markdown(
"""**Netværkstæthed** angiver, hvor stor en andel af alle mulige forbindelser, 
der faktisk forekommer i netværket. En høj tæthed betyder, at mange grupper 
samarbejder med hinanden, mens en lav tæthed afspejler et mere spredt og 
opdelt samarbejdsmønster.

Vær opmærksom på valg af filtre i sidepanelet samt intra- og interfakultetskanter.
""")

        st.metric("Netværkstæthed",f"{density:.3f}")

        colA, colB, colC = st.columns(3)
        with colA:
            st.metric(
                "Modularitet (foruddefinerede klynger)", 
                f"{modularity_pre:.3f}" if not np.isnan(modularity_pre) else "n/a",
                help = f"Klyngeniveau: {len(communities)} {abbrs.get(comm_key)}"
                )

        with colB:
            st.metric("Modularitet (greedy)",
            f"{modularity_greedy:.3f}" if not np.isnan(modularity_greedy) else "n/a",
            help = "Greedy algoritmen opdeler netværket i optimale klynger"
            )
        
        with colC:
            st.metric("Antal greedy-klynger", n_comms)

        # Tabelvisning
        table_rows = []

        # Prædefinerede klynger
        for group, members in zip(sorted(seen_groups), communities):
            table_rows.append({
                "Klynge (defineret)": group,
                "Antal noder": len(members),
                "Type": comm_key
            })
        
        schema = [
            ("Klynge (defineret)", pa.string()),
            ("Antal noder", pa.int64()),
            ("Type", pa.string())
        ]

        table = build_table(table_rows, schema)
        with st.expander("Se detaljeret klyngetabel med prædefinerede klynger"):
            st.dataframe(table, hide_index = True, width = "stretch")
            csv_bytes = rows_to_csv_bytes(table_rows, [name for name, _ in schema])
            st.download_button("Download (.csv)", data = csv_bytes, file_name = f"klynger_{mode}_{year}.csv")

        # Greedy klynger
        table_rows = []
        for i, comm in enumerate(greedy_comms, start = 1):
            table_rows.append({
                "Klynge (defineret)": f"Greedy {i}",
                "Antal noder": len(comm),
                "Type": "greedy"
            })

        schema = [
            ("Klynge (defineret)", pa.string()),
            ("Antal noder", pa.int64()),
            ("Type", pa.string())
        ]

        table = build_table(table_rows, schema)

        with st.expander("Se detaljeret klyngetabel med greedy algoritme"):
            st.dataframe(table, hide_index = True, width = "stretch")
            csv_bytes = rows_to_csv_bytes(table_rows, [name for name, _ in schema])
            st.download_button("Download (.csv)", data = csv_bytes, file_name = f"greedy_klynger_{mode}_{year}.csv")







if "Centralitet" in tabs_dict:
    with tabs_dict["Centralitet"]:
        st.subheader("Centralitet for udsnittet")

        st.markdown("""**Centralitet** er et mål for, hvor *vigtige* noderne 
(fakulteter, institutter eller stillingsgrupper) er i det valgte udsnit af 
sampubliceringsnetværket. To mål vises:  

**Weighted degree (styrke)** er den samlede vægt af alle publikationer, som et fakultet, institut
eller en stillingsgruppe indgår i:

- En høj værdi betyder, at noden deltager i mange eller store sampubliceringer.
- Det er et mål for aktivitet, ikke nødvendigvis strategisk position.

**Betweenness centralitet (brobyggerrolle)** måler, hvor ofte en node ligger på den *korteste sti* mellem to andre noder.

- En høj værdi betyder, at noden fungerer som en bro mellem grupper, der ellers ikke er direkte forbundne.
- Det indikerer strukturel betydning: noder, der binder netværket sammen og skaber forbindelser på tværs.
""")

        if faculty_wd_sorted:
            with st.expander("Se fuld tabel over fakulteters centralitet"):
                st.markdown("Kan der stå noget her?")

                fac_rows = []

                wd_map = {k: v for k, v in faculty_wd_sorted} if faculty_wd_sorted else {}
                bs_map = {k: v for k, v in faculty_bs_sorted} if faculty_bs_sorted else {}

                all_facs = sorted(set(list(wd_map.keys()) + list(bs_map.keys())))
                for f in all_facs:
                    fac_rows.append({
                        "Fakultet": f,
                        "Weighted degree": float(wd_map.get(f, 0.0)),
                        "Betweenness": float(bs_map.get(f, 0.0)),
                        "mode": mode
                    })
                
                # Schema og tabel
                fac_schema = [
                    ("Fakultet", pa.string()),
                    ("Weighted degree", pa.float64()),
                    ("Betweenness", pa.float64()),
                    ("mode", pa.string())
                ]
                fac_table = build_table(fac_rows, fac_schema) if fac_rows else None

                # UI
                if fac_rows:
                    st.dataframe(
                        fac_table,
                        width = "stretch",
                        hide_index = True
                    )
                    csv_bytes = rows_to_csv_bytes(fac_rows, [name for name, _ in fac_schema])
                    st.download_button(
                        "Download (.csv)",
                        data = csv_bytes,
                        file_name = f"centralitet_fakulteter_{year}_{mode}.csv",
                        mime = "text/csv"
                    )
                else:
                    st.info("Ingen fakultetsrækker i udsnittet")

        if inst_wd_sorted:
            with st.expander("Se fuld tabel over institutters centralitet"):
                inst_rows = []

                wd_map = {k: v for k, v in inst_wd_sorted} if inst_wd_sorted else {}
                bs_map = {k: v for k, v in inst_bs_sorted} if inst_bs_sorted else {}

                inst_to_fac = {}
                for nid, meta in node_meta.items():
                    inst = meta.get("inst", "")
                    fac = meta.get("fac", "")
                    if inst and fac and inst not in inst_to_fac:
                        inst_to_fac[inst] = fac
                
                all_insts = sorted(set(list(wd_map.keys()) + list(bs_map.keys())))
                for inst in all_insts:
                    inst_rows.append({
                        "Institut": inst,
                        "Fakultet": inst_to_fac.get(inst, ""),
                        "Weighted degree": float(wd_map.get(inst, 0.0)),
                        "Betweenness": float(bs_map.get(inst, 0.0)),
                        "mode": mode
                    })
                
                inst_schema = [
                    ("Institut", pa.string()),
                    ("Fakultet", pa.string()),
                    ("Weighted degree", pa.float64()),
                    ("Betweenness", pa.float64()),
                    ("mode", pa.string())
                ]
                inst_table = build_table(inst_rows, inst_schema) if inst_rows else None

                if inst_rows:
                    st.dataframe(inst_table, width = "stretch", hide_index = True)
                    csv_bytes = rows_to_csv_bytes(inst_rows, [name for name, _ in inst_schema])
                    st.download_button(
                        "Download (.csv)",
                        data = csv_bytes,
                        file_name = f"centralitet_institutter_{year}_{mode}.csv"

                    )

        if grp_wd_sorted:
            with st.expander("Se fuld tabel over stillinggruppers centralitet"):
                rows, schema = build_grp_table_by_mode(
                    weighted_deg = grp_wd_sorted,
                    bet_cent = grp_bs_sorted,
                    node_meta = node_meta,
                    mode = mode,
                )

                if rows:
                    table = build_table(rows, schema)
                    st.dataframe(table, width = "stretch", hide_index = True)
                    csv_bytes = rows_to_csv_bytes(rows, [name for name, _ in schema])
                    st.download_button(
                        "Download (.csv)",
                        data = csv_bytes,
                        file_name = f"centralitet_stillingsgrupper_{year}_{mode}.csv"
                    )


if "Datagrundlag" in tabs_dict:
    with tabs_dict["Datagrundlag"]:
        st.subheader("Datagrundlag")

        curis_created_ts = os.path.getctime("H:/Sampubliceringsanalyse/Data/KU_titles_CURIS_03.csv")
        curis_created = datetime.datetime.fromtimestamp(curis_created_ts).strftime("%Y-%m-%d")

        hr_created_ts = os.path.getctime("H:/Sampubliceringsanalyse/Data/Personalesammensætning.csv")
        hr_created = datetime.datetime.fromtimestamp(hr_created_ts).strftime("%Y-%m-%d")

        st.markdown(
f"""Netværk og tilhørende analyse bygger på:
- CURIS-publikationer for 2021-2025, udtrukket d. {curis_created}. Bemærk: data for
2025 er ufuldstændige som følge af udtræksdatoen.
- VIP-forfattere matchet til stillingsgrupper, institutter og fakulteter via HR-data, udtrukket d. {hr_created}.
- Kontruktion af det vægtede publikationsnetværk baseret på medforfatterskaber.

Hver node i netværkerne repræsenterer en kombination af stillingsgruppe og 
organisatiorisk enhed. Hver kant angiver den vægtede sampublicering mellem disse grupper.  
"""
)

        st.markdown("""
**Publikationspraksis**

Da publikationerne indgår i et netværk baseret på medforfatterskaber, 
er kun publikationer med mindst to forfattere fra forskellige stillingsgrupper, 
fakulteter eller institutter blevet inkluderet. Publikationer med én forfatter
indgår derfor ikke i netværket. 
""")

        counts = defaultdict(lambda: defaultdict(lambda: {"0": 0, "1": 0, "gt1": 0}))
        pub_types = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        csv_fil = "H:/Sampubliceringsanalyse/Data/KU_titles_CURIS_VIP_data.csv"

        with open(csv_fil, encoding = "utf-8") as f:
            reader = csv.DictReader(f, delimiter = ";")
            for row in reader:
                pub_year = int(row["Udgivelsesår"])
                if pub_year != year:
                    continue

                pub_type = row["Type"].strip()

                raw_auth = row["Forfatter(e)"].strip()
                if raw_auth in ("0", "['0']", "[[0]]", "[['0']]"):
                    key = "0"
                else:
                    authors = [a.strip() for a in raw_auth.split(",") if a.strip()]
                    if len(authors) == 1:
                        key = "1"
                    else:
                        key = "gt1"
                        
                raw_fac = row["Fakultet(er)"]
                clean = raw_fac.replace("[", "").replace("]", "").replace("'","")
                fac_list = [f.strip() for f in clean.split(",") if f.strip() and f.strip() != "0"]

                if not fac_list:
                    continue

                for fac in fac_list:
                    counts[fac][year][key] += 1
                    pub_types[fac][year][pub_type] += 1
        
        all_pub_types = sorted({t for fac in pub_types for y in pub_types[fac] for t in pub_types[fac][y]})
        faculties = sorted(counts.keys())

        zero_counts = [counts[fac][year]["0"] for fac in faculties]
        one_counts = [counts[fac][year]["1"] for fac in faculties]
        many_counts = [counts[fac][year]["gt1"] for fac in faculties]

        ratio_per_fac = {}
        for fac, one, many in zip(faculties, one_counts, many_counts):
            if (one + many) > 0:
                ratio_per_fac[fac] = one / (one + many ) * 100
            else:
                ratio_per_fac[fac] = None
        
        valid_ratios = {fac: v for fac, v in ratio_per_fac.items() if v is not None}

        if valid_ratios:
            min_fac = min(ratio_per_fac, key = ratio_per_fac.get)
            min_val = ratio_per_fac[min_fac]

            max_fac = max(ratio_per_fac, key = ratio_per_fac.get)
            max_val = ratio_per_fac[max_fac]

            abs_one = counts.get(max_fac, {}).get(year, {}).get("1", 0)
        
        else:
            min_fac, min_val, max_fac, max_val = None
            abs_ons = 0


        #abs_one = counts[max_fac][year]["1"]

        if min_fac is not None and max_fac is not None:
            st.markdown(
f"""I figuren nedenfor vises publikationspoduktionen for hvert fakultet i **{year}**.
Publikationerne er opdelte efter antal forfattere, således at hver søjle angiver
andelen af hhv. publikationer med én forfatter og publikationer med
to eller flere forfattere. 

Andelen af publikationer med én forfatter varierer betydelige mellem fakulteterne; 
fra **{min_val:.1f}%** på **{min_fac}** til **{max_val:.1f}%** på **{max_fac}**. 
Altså, er {max_val:.1f}% af {max_fac}s publikationer ikke medregnet i netværksanalysen,
svarende til **{abs_one}** frasorterede publikationer.

Det er et gennemgående mønster over årene, at visse fakulteter har en høj andel af
publikationer med flere forfattere - og er dermed mindre sårbare overfor årlige udsving
i publikationsmængden.
""")
    
    
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x = faculties,
            y = one_counts,
            name = "1 forfatter",
        ))

        fig.add_trace(go.Bar(
            x = faculties,
            y = many_counts,
            name = ">1 forfatter",
        ))
        
        fig.update_layout(
            barmode = "stack",
            yaxis_title = "Antal publikationer",
        )

        st.plotly_chart(fig, width = "stretch")


        st.markdown(f"""
Som det fremgår af figuren nedenfor, er der tydelig sammenhæng mellem fakulteternes
samarbejdsmønstre og deres valg af publikationstyper. Fakulteter med en lav andel 
af publikationer med flere forfattere har typisk også en lavere andel af 
tidsskriftsartikler og en større spredning på andre publiceringsformater 
(f.eks. bog- og avisbidrag, monografier og antologier). 

Dette peger på, at forskelle i publikationstyper kan være med at til forklare
variationerne i andelen af flerforfatterskaber på tværs af fakulteterne. Fagområder
med en bredere medieprofil vil ofte have en højere forekomst af "en forfatter"-
publikationer, mens fakulteter med en stærk tidsskrifttradition typisk har en 
større andel af publikationer med flere forfattere.

""")


        fig = go.Figure()
        
        for i, t in enumerate(all_pub_types):
            y_vals = [pub_types[fac][year][t] for fac in faculties]

            fig.add_trace(go.Bar(
                x = faculties,
                y = y_vals,
                name = t,
            ))

        fig.update_layout(
            barmode = "stack",
            yaxis_title = "Antal publikationer"
        )

        st.plotly_chart(fig, width = "stretch")









st.markdown("""
<hr style="margin-top: 50px;">
<div style="text-align:center; color:#666; font-size: 0.9em;">
  KU Analyse · Amanda Schramm Petersen · <a href="mailto:ascp@adm.ku.dk">ascp@adm.ku.dk</a>
</div>
""", unsafe_allow_html=True)
