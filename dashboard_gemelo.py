"""
dashboard_gemelo.py
===================
Gemelo Digital — Panadería Dora del Hoyo  v2.0
================================================
Dashboard profesional Dash con:
  • Demanda histórica con 4 gráficos (barras, heatmap, tendencia, treemap)
  • Planeación Agregada PuLP/CBC con alertas automáticas
  • Desagregación por producto con tablas individuales
  • Simulación SimPy con Gantt y colas
  • KPIs: throughput, lead time, WIP, takt time, cumplimiento, radar
  • Sensores virtuales: temperatura, humedad, ocupación, cola
  • 8 Escenarios What-If con tabla resumen comparativa
  • Config condicional por tab (sin parámetros irrelevantes)
  • UI dark navy industrial premium — Syne + JetBrains Mono

INSTALACIÓN:
    pip install dash dash-bootstrap-components simpy pulp pandas numpy plotly

EJECUCIÓN:
    python dashboard_gemelo.py
    → http://127.0.0.1:8050
"""

import math, random, threading, warnings
import numpy as np
import pandas as pd
import simpy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, PULP_CBC_CMD
import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# DATOS MAESTROS
# ─────────────────────────────────────────────────────────────────────────────
PRODUCTOS  = ["Brownies","Mantecadas","Mantecadas_Amapola","Torta_Naranja","Pan_Maiz"]
MESES      = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
MESES_FULL = ["January","February","March","April","May","June",
              "July","August","September","October","November","December"]

DEM_HISTORICA = {
    "Brownies":           [315,804,734,541,494, 59,315,803,734,541,494, 59],
    "Mantecadas":         [125,780,432,910,275, 68,512,834,690,455,389,120],
    "Mantecadas_Amapola": [320,710,520,251,631,150,330,220,710,610,489,180],
    "Torta_Naranja":      [100,250,200,101,190, 50,100,220,200,170,180,187],
    "Pan_Maiz":           [330,140,143, 73, 83, 48, 70, 89,118, 83, 67, 87],
}
HORAS_PRODUCTO = {
    "Brownies":0.866,"Mantecadas":0.175,"Mantecadas_Amapola":0.175,
    "Torta_Naranja":0.175,"Pan_Maiz":0.312,
}
INV_INICIAL    = {p: 0 for p in PRODUCTOS}
TAMANO_LOTE    = {"Brownies":12,"Mantecadas":10,"Mantecadas_Amapola":10,"Torta_Naranja":12,"Pan_Maiz":15}
CAPACIDAD_BASE = {"mezcla":2,"dosificado":2,"horno":3,"enfriamiento":4,"empaque":2,"amasado":1}

RUTAS = {
    "Brownies":           [("Mezclado","mezcla",12,18),("Moldeado","dosificado",8,14),
                           ("Horneado","horno",30,40),("Enfriamiento","enfriamiento",25,35),
                           ("Corte_Empaque","empaque",8,12)],
    "Mantecadas":         [("Mezclado","mezcla",12,18),("Dosificado","dosificado",16,24),
                           ("Horneado","horno",20,30),("Enfriamiento","enfriamiento",35,55),
                           ("Empaque","empaque",4,6)],
    "Mantecadas_Amapola": [("Mezclado","mezcla",12,18),("Inc_Semillas","mezcla",8,12),
                           ("Dosificado","dosificado",16,24),("Horneado","horno",20,30),
                           ("Enfriamiento","enfriamiento",36,54),("Empaque","empaque",4,6)],
    "Torta_Naranja":      [("Mezclado","mezcla",16,24),("Dosificado","dosificado",8,12),
                           ("Horneado","horno",32,48),("Enfriamiento","enfriamiento",48,72),
                           ("Desmolde","dosificado",8,12),("Empaque","empaque",8,12)],
    "Pan_Maiz":           [("Mezclado","mezcla",12,18),("Amasado","amasado",16,24),
                           ("Moldeado","dosificado",12,18),("Horneado","horno",28,42),
                           ("Enfriamiento","enfriamiento",36,54),("Empaque","empaque",4,6)],
}
PARAMS_AGRE = {
    "Ct":4310,"Ht":100000,"PIt":100000,"CRt":11364,"COt":14205,
    "CW_mas":14204,"CW_menos":15061,"M":1,"LR_inicial":44*4*10,"inv_seg":0.0,
}
PROD_COLORS = {
    "Brownies":"#F4A261","Mantecadas":"#48CAE4","Mantecadas_Amapola":"#95D5B2",
    "Torta_Naranja":"#C77DFF","Pan_Maiz":"#FF6B6B",
}
PROD_ICONS = {
    "Brownies":"🍫","Mantecadas":"🧁","Mantecadas_Amapola":"🌸",
    "Torta_Naranja":"🍊","Pan_Maiz":"🌽",
}

# ─────────────────────────────────────────────────────────────────────────────
# MOTOR DE CÁLCULO
# ─────────────────────────────────────────────────────────────────────────────
def _dem_horas(factor=1.0):
    return {mes: round(sum(DEM_HISTORICA[p][i]*HORAS_PRODUCTO[p] for p in PRODUCTOS)*factor, 4)
            for i, mes in enumerate(MESES_FULL)}

def run_agregacion(dem_horas, params=None):
    p = params or PARAMS_AGRE.copy()
    meses = MESES_FULL
    mdl = LpProblem("Agr", LpMinimize)
    P   = LpVariable.dicts("P",  meses, lowBound=0)
    I   = LpVariable.dicts("I",  meses, lowBound=0)
    S   = LpVariable.dicts("S",  meses, lowBound=0)
    LR  = LpVariable.dicts("LR", meses, lowBound=0)
    LO  = LpVariable.dicts("LO", meses, lowBound=0)
    LU  = LpVariable.dicts("LU", meses, lowBound=0)
    NI  = LpVariable.dicts("NI", meses)
    Wm  = LpVariable.dicts("Wm", meses, lowBound=0)
    Wd  = LpVariable.dicts("Wd", meses, lowBound=0)
    mdl += lpSum(p["Ct"]*P[t]+p["Ht"]*I[t]+p["PIt"]*S[t]+p["CRt"]*LR[t]+p["COt"]*LO[t]
                 +p["CW_mas"]*Wm[t]+p["CW_menos"]*Wd[t] for t in meses)
    for idx, t in enumerate(meses):
        d = dem_horas[t]; tp = meses[idx-1] if idx > 0 else None
        if idx == 0: mdl += NI[t] == P[t]-d
        else:        mdl += NI[t] == NI[tp]+P[t]-d
        mdl += NI[t] == I[t]-S[t]
        mdl += LU[t]+LO[t] == p["M"]*P[t]
        mdl += LU[t] <= LR[t]
        if idx == 0: mdl += LR[t] == p["LR_inicial"]+Wm[t]-Wd[t]
        else:        mdl += LR[t] == LR[tp]+Wm[t]-Wd[t]
    mdl.solve(PULP_CBC_CMD(msg=False))
    costo = value(mdl.objective)
    ini_l, fin_l = [], []
    for idx, t in enumerate(meses):
        ini = 0.0 if idx == 0 else fin_l[-1]
        ini_l.append(ini); fin_l.append(ini+P[t].varValue-dem_horas[t])
    df = pd.DataFrame({
        "Mes": MESES,
        "Demanda_HH":       [round(dem_horas[t],2)    for t in meses],
        "Produccion_HH":    [round(P[t].varValue,2)   for t in meses],
        "Backlog_HH":       [round(S[t].varValue,2)   for t in meses],
        "Horas_Regulares":  [round(LR[t].varValue,2)  for t in meses],
        "Horas_Extras":     [round(LO[t].varValue,2)  for t in meses],
        "Inv_Inicial_HH":   [round(v,2)               for v in ini_l],
        "Inv_Final_HH":     [round(v,2)               for v in fin_l],
        "Contratacion":     [round(Wm[t].varValue,2)  for t in meses],
        "Despidos":         [round(Wd[t].varValue,2)  for t in meses],
    })
    return df, costo

def run_desagregacion(prod_hh, factor=1.0):
    meses = MESES_FULL
    mdl = LpProblem("Desag", LpMinimize)
    X = {(p,t): LpVariable(f"X_{p}_{t}", lowBound=0) for p in PRODUCTOS for t in meses}
    I = {(p,t): LpVariable(f"I_{p}_{t}", lowBound=0) for p in PRODUCTOS for t in meses}
    S = {(p,t): LpVariable(f"S_{p}_{t}", lowBound=0) for p in PRODUCTOS for t in meses}
    mdl += lpSum(100000*I[p,t]+150000*S[p,t] for p in PRODUCTOS for t in meses)
    for idx, t in enumerate(meses):
        tp = meses[idx-1] if idx > 0 else None
        mdl += lpSum(HORAS_PRODUCTO[p]*X[p,t] for p in PRODUCTOS) <= prod_hh[t]
        for pr in PRODUCTOS:
            d = int(DEM_HISTORICA[pr][idx]*factor)
            if idx == 0: mdl += I[pr,t]-S[pr,t] == INV_INICIAL[pr]+X[pr,t]-d
            else:        mdl += I[pr,t]-S[pr,t] == I[pr,tp]-S[pr,tp]+X[pr,t]-d
    mdl.solve(PULP_CBC_CMD(msg=False))
    out = {}
    for pr in PRODUCTOS:
        rows = []
        for idx, t in enumerate(meses):
            xv  = round(X[pr,t].varValue or 0, 2)
            iv  = round(I[pr,t].varValue or 0, 2)
            sv  = round(S[pr,t].varValue or 0, 2)
            ini = INV_INICIAL[pr] if idx==0 else round(I[pr,meses[idx-1]].varValue or 0, 2)
            rows.append({"Mes":MESES[idx], "Demanda":int(DEM_HISTORICA[pr][idx]*factor),
                         "Produccion":xv, "Inv_Ini":ini, "Inv_Fin":iv, "Backlog":sv})
        out[pr] = pd.DataFrame(rows)
    return out

def run_simulacion(plan_und, cap_rec=None, falla=False, factor_t=1.0, semilla=42):
    random.seed(semilla); np.random.seed(semilla)
    if cap_rec is None: cap_rec = CAPACIDAD_BASE.copy()
    lotes_data, uso_rec, sensores = [], [], []
    dur_mes = 44*4*60

    def reg(env, recursos, prod=""):
        ts = round(env.now, 3)
        for nm, r in recursos.items():
            uso_rec.append({"tiempo":ts,"recurso":nm,"ocupados":r.count,
                            "cola":len(r.queue),"capacidad":r.capacity,"producto":prod})

    def sensor_horno(env, recursos):
        while True:
            ocp   = recursos["horno"].count
            temp  = round(np.random.normal(160+ocp*20, 5), 2)
            humid = round(np.random.normal(45-ocp*5, 3), 2)
            sensores.append({"tiempo":round(env.now,1),"temperatura":temp,"humedad":humid,
                             "horno_ocup":ocp,"horno_cola":len(recursos["horno"].queue)})
            yield env.timeout(10)

    def proceso_lote(env, lid, prod, tam, recursos):
        t0 = env.now; esperas = {}
        for etapa, rec_nm, tmin, tmax in RUTAS[prod]:
            escala = math.sqrt(tam / TAMANO_LOTE[prod])
            tp = random.uniform(tmin, tmax) * escala * factor_t
            if falla and rec_nm == "horno": tp += random.uniform(10, 30)
            reg(env, recursos, prod)
            t_ei = env.now
            with recursos[rec_nm].request() as req:
                yield req
                esperas[etapa] = round(env.now-t_ei, 3)
                reg(env, recursos, prod)
                yield env.timeout(tp)
            reg(env, recursos, prod)
        lotes_data.append({
            "lote_id":lid, "producto":prod, "tamano":tam,
            "t_creacion":round(t0,3), "t_fin":round(env.now,3),
            "tiempo_sistema":round(env.now-t0,3),
            "total_espera":round(sum(esperas.values()),3),
        })

    env      = simpy.Environment()
    recursos = {nm: simpy.Resource(env, capacity=cap) for nm, cap in cap_rec.items()}
    env.process(sensor_horno(env, recursos))
    lotes=[]; ctr=[0]
    for prod, unid in plan_und.items():
        if unid <= 0: continue
        tam = TAMANO_LOTE[prod]; n = math.ceil(unid/tam)
        tasa = dur_mes/max(n,1); ta = random.expovariate(1/max(tasa,1)); rem = unid
        for _ in range(n):
            lotes.append((round(ta,2), prod, min(tam,int(rem))))
            rem -= tam; ta += random.expovariate(1/max(tasa,1))
    lotes.sort(key=lambda x: x[0])

    def lanzador():
        for ta, prod, tam in lotes:
            yield env.timeout(max(ta-env.now, 0))
            lid = f"{prod[:3].upper()}_{ctr[0]:04d}"; ctr[0]+=1
            env.process(proceso_lote(env, lid, prod, tam, recursos))
    env.process(lanzador())
    env.run(until=dur_mes*1.8)
    df_l = pd.DataFrame(lotes_data) if lotes_data else pd.DataFrame()
    df_u = pd.DataFrame(uso_rec)    if uso_rec    else pd.DataFrame()
    df_s = pd.DataFrame(sensores)   if sensores   else pd.DataFrame()
    return df_l, df_u, df_s

def calc_utilizacion(df_u):
    if df_u.empty: return pd.DataFrame()
    filas = []
    for rec, grp in df_u.groupby("recurso"):
        grp = grp.sort_values("tiempo").reset_index(drop=True)
        cap = grp["capacidad"].iloc[0]; t = grp["tiempo"].values; ocp = grp["ocupados"].values
        fn  = np.trapezoid if hasattr(np,"trapezoid") else np.trapz
        util = round(fn(ocp,t)/(cap*(t[-1]-t[0]))*100,2) if len(t)>1 and (t[-1]-t[0])>0 else 0.0
        filas.append({"Recurso":rec, "Utilización_%":util,
                      "Cola Prom":round(grp["cola"].mean(),3),
                      "Cola Máx":int(grp["cola"].max()),
                      "Capacidad":int(cap),
                      "Cuello Botella":util>=80 or grp["cola"].mean()>0.5})
    return pd.DataFrame(filas).sort_values("Utilización_%",ascending=False).reset_index(drop=True)

def calc_kpis(df_l, plan):
    if df_l.empty: return pd.DataFrame()
    dur = (df_l["t_fin"].max()-df_l["t_creacion"].min())/60
    rows = []
    for pr in PRODUCTOS:
        sub = df_l[df_l["producto"]==pr]
        if sub.empty: continue
        und = sub["tamano"].sum(); plan_und = plan.get(pr,0)
        tp  = round(und/max(dur,0.01),3)
        ct  = round((sub["tiempo_sistema"]/sub["tamano"]).mean(),3)
        lt  = round(sub["tiempo_sistema"].mean(),3)
        dem_avg = sum(DEM_HISTORICA[pr])/12
        takt = round((44*4*60)/max(dem_avg/TAMANO_LOTE[pr],1),2)
        wip  = round(tp*(lt/60),2)
        rows.append({"Producto":pr, "Und Producidas":int(und), "Plan":int(plan_und),
                     "Throughput (und/h)":tp, "Cycle Time (min/und)":ct,
                     "Lead Time (min/lote)":lt, "WIP":wip, "Takt Time (min/lote)":takt,
                     "Cumplimiento %":round(min(und/max(plan_und,1)*100,100),2)})
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# TEMA & TOKENS
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":      "#050A13",
    "card":    "#0A1628",
    "panel":   "#0F1F3D",
    "border":  "#1E3A5F",
    "accent":  "#F4A261",
    "a2":      "#48CAE4",
    "a3":      "#95D5B2",
    "text":    "#E2E8F0",
    "muted":   "#64748B",
    "dim":     "#334155",
    "ok":      "#10B981",
    "warn":    "#F59E0B",
    "err":     "#EF4444",
}
THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", color="#94A3B8", size=11),
    xaxis=dict(gridcolor="#1E293B", zerolinecolor="#334155"),
    yaxis=dict(gridcolor="#1E293B", zerolinecolor="#334155"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    margin=dict(l=50,r=20,t=60,b=50),
    colorway=["#F4A261","#48CAE4","#95D5B2","#C77DFF","#FF6B6B","#FBBF24"],
)
CARD = {"background":C["card"],"border":f"1px solid {C['border']}",
        "borderRadius":"12px","padding":"20px","boxShadow":"0 4px 24px rgba(0,0,0,.4)"}
LABEL = {"color":C["muted"],"fontSize":"10px","fontFamily":"JetBrains Mono, monospace",
         "letterSpacing":"0.15em","textTransform":"uppercase","marginBottom":"6px","display":"block"}
INP   = {"background":"#061020","color":C["text"],"border":f"1px solid {C['border']}",
         "borderRadius":"8px","fontFamily":"JetBrains Mono, monospace","fontSize":"12px"}

def apply_theme(fig, title="", h=400):
    fig.update_layout(**THEME, height=h,
        title=dict(text=title, x=0.5,
                   font=dict(size=15, color=C["accent"], family="Syne, sans-serif")))
    fig.update_xaxes(gridcolor="#1E293B", zerolinecolor="#334155")
    fig.update_yaxes(gridcolor="#1E293B", zerolinecolor="#334155")
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def kpi(title, val, unit="", color=None, icon="◈", sub=None):
    color = color or C["accent"]
    return html.Div([
        html.Div([html.Span(icon,style={"color":color,"marginRight":"6px"}),
                  html.Span(title,style={**LABEL,"display":"inline","marginBottom":"0"})],
                 style={"display":"flex","alignItems":"center","marginBottom":"8px"}),
        html.Div([
            html.Span(str(val),style={"fontSize":"26px","fontWeight":"800","color":color,
                                       "fontFamily":"Syne, sans-serif","lineHeight":"1"}),
            html.Span(f" {unit}",style={"fontSize":"10px","color":C["muted"],"marginLeft":"4px",
                                         "fontFamily":"JetBrains Mono"}),
        ]),
        html.Div(sub,style={"fontSize":"10px","color":C["muted"],"marginTop":"4px",
                             "fontFamily":"JetBrains Mono"}) if sub else None,
    ], style={**CARD,"flex":"1","minWidth":"145px",
              "background":f"linear-gradient(135deg,{C['card']},{C['panel']})",
              "borderLeft":f"3px solid {color}"})

def sec(title, sub="", badge=None):
    return html.Div([
        html.Div([
            html.Div([
                html.Span(title,style={"fontFamily":"Syne, sans-serif","fontWeight":"800",
                                        "fontSize":"20px","color":C["text"]}),
                html.Span(badge,style={"background":C["accent"],"color":C["bg"],
                                        "fontSize":"9px","fontFamily":"JetBrains Mono","fontWeight":"700",
                                        "padding":"2px 8px","borderRadius":"20px","marginLeft":"10px",
                                        "verticalAlign":"middle"}) if badge else None,
            ], style={"display":"flex","alignItems":"center"}),
            html.Div(sub,style={"color":C["muted"],"fontSize":"11px",
                                 "fontFamily":"JetBrains Mono","marginTop":"4px"}) if sub else None,
        ]),
        html.Div(style={"height":"2px","background":f"linear-gradient(90deg,{C['accent']},transparent)",
                         "marginTop":"10px","borderRadius":"2px"}),
    ], style={"marginBottom":"22px"})

def dtable(df, tid, ps=12):
    if df is None or df.empty:
        return html.Div("Sin datos",style={"color":C["muted"],"padding":"20px"})
    return dash_table.DataTable(
        id=tid, columns=[{"name":c,"id":c} for c in df.columns],
        data=df.round(3).to_dict("records"), page_size=ps,
        style_table={"overflowX":"auto","borderRadius":"8px","overflow":"hidden"},
        style_header={"backgroundColor":"#061020","color":C["accent"],
                       "fontFamily":"JetBrains Mono","fontSize":"10px",
                       "border":f"1px solid {C['border']}","letterSpacing":"0.1em",
                       "fontWeight":"600","padding":"10px 14px"},
        style_cell={"backgroundColor":C["card"],"color":"#94A3B8",
                     "fontFamily":"JetBrains Mono","fontSize":"11px",
                     "border":f"1px solid {C['panel']}","padding":"8px 14px","textAlign":"right"},
        style_data_conditional=[
            {"if":{"row_index":"odd"},"backgroundColor":"#0A1020"},
            {"if":{"state":"selected"},"backgroundColor":"#1E3A5F44"},
        ],
    )

def alert(text, kind="info"):
    cols = {"info":C["a2"],"warn":C["warn"],"ok":C["ok"],"err":C["err"]}
    col  = cols.get(kind, C["a2"])
    ico  = {"info":"ℹ","warn":"⚠","ok":"✓","err":"✗"}.get(kind,"ℹ")
    return html.Div([
        html.Span(ico,style={"marginRight":"8px","fontWeight":"bold"}),
        html.Span(text,style={"fontFamily":"JetBrains Mono","fontSize":"11px"}),
    ], style={"background":f"{col}18","border":f"1px solid {col}44",
               "borderLeft":f"3px solid {col}","borderRadius":"6px",
               "padding":"10px 16px","color":col,"marginBottom":"10px"})

def run_btn():
    return html.Button(
        [html.Span("▶ ",style={"marginRight":"4px"}),html.Span("EJECUTAR PIPELINE")],
        id="btn-run", n_clicks=0,
        style={"background":"linear-gradient(135deg,#1D4ED8,#2563EB)","color":"white",
               "border":"none","padding":"10px 28px","fontFamily":"Syne, sans-serif",
               "fontWeight":"700","fontSize":"13px","letterSpacing":"0.1em",
               "borderRadius":"8px","cursor":"pointer","boxShadow":"0 4px 16px rgba(37,99,235,.4)"})

def status_div():
    return html.Div(id="run-status",style={"fontSize":"11px","fontFamily":"JetBrains Mono",
                                            "padding":"8px 0","color":C["ok"]})

def pslider(label, sid, mn, mx, step, val, marks=None):
    return html.Div([
        html.Span(label,style=LABEL),
        dcc.Slider(id=sid,min=mn,max=mx,step=step,value=val,
                   marks=marks or {mn:str(mn),mx:str(mx)},
                   tooltip={"placement":"top","always_visible":True}),
    ], style={"marginBottom":"4px"})

# ─────────────────────────────────────────────────────────────────────────────
# FIGURAS
# ─────────────────────────────────────────────────────────────────────────────
def f_barras():
    fig = go.Figure()
    for p in PRODUCTOS:
        fig.add_trace(go.Bar(x=MESES, y=DEM_HISTORICA[p],
                              name=f"{PROD_ICONS[p]} {p.replace('_',' ')}",
                              marker_color=PROD_COLORS[p], opacity=0.85,
                              hovertemplate=f"<b>{p}</b><br>%{{x}}<br><b>%{{y:,.0f}}</b> und<extra></extra>"))
    fig.update_layout(**THEME, barmode="group", height=400,
        title=dict(text="Demanda Histórica por Producto & Mes",x=0.5,
                   font=dict(size=15,color=C["accent"],family="Syne, sans-serif")),
        xaxis_title="Mes", yaxis_title="Unidades",
        legend=dict(bgcolor="rgba(0,0,0,0)",orientation="h",y=-0.28,x=0.5,xanchor="center"))
    return fig

def f_heatmap():
    z = [[DEM_HISTORICA[p][i] for i in range(12)] for p in PRODUCTOS]
    y = [f"{PROD_ICONS[p]} {p.replace('_',' ')}" for p in PRODUCTOS]
    annotations = [dict(x=j,y=i,text=f"{v:,}",showarrow=False,
                        font=dict(size=9,color="white" if v>400 else "#111",family="JetBrains Mono"))
                   for i,row in enumerate(z) for j,v in enumerate(row)]
    fig = go.Figure(go.Heatmap(z=z, x=MESES, y=y,
        colorscale=[[0,"#061020"],[0.35,"#1E3A5F"],[0.7,"#F4A261"],[1,"#FF6B6B"]],
        hovertemplate="%{y}<br>%{x}<br><b>%{z:,.0f}</b> und<extra></extra>",
        colorbar=dict(tickcolor=C["muted"],tickfont=dict(size=9))))
    fig.update_layout(annotations=annotations)
    apply_theme(fig,"Mapa de Calor — Estacionalidad",320)
    return fig

def f_tendencia():
    total = [sum(DEM_HISTORICA[p][i] for p in PRODUCTOS) for i in range(12)]
    ma    = pd.Series(total).rolling(3,center=True).mean().tolist()
    fig   = go.Figure()
    fig.add_trace(go.Bar(x=MESES, y=total, name="Demanda Total",
        marker=dict(color=total,colorscale=[[0,"#1E3A5F"],[0.5,"#F4A261"],[1,"#FF6B6B"]],showscale=False),
        hovertemplate="<b>%{x}</b><br>Total: %{y:,.0f} und<extra></extra>"))
    fig.add_trace(go.Scatter(x=MESES, y=ma, mode="lines", name="Media móvil 3m",
        line=dict(color=C["a2"],width=2.5,dash="dot")))
    apply_theme(fig,"Demanda Total Consolidada + Tendencia",320)
    fig.update_layout(xaxis_title="Mes",yaxis_title="Unidades",
        legend=dict(orientation="h",y=-0.22,x=0.5,xanchor="center"))
    return fig

def f_treemap():
    totales = {p:sum(DEM_HISTORICA[p]) for p in PRODUCTOS}
    total_g = sum(totales.values())
    labels  = [f"{PROD_ICONS[p]} {p.replace('_',' ')}<br>{totales[p]:,} und ({totales[p]/total_g*100:.0f}%)"
               for p in PRODUCTOS]
    fig = go.Figure(go.Treemap(
        labels=labels, parents=[""]*len(PRODUCTOS),
        values=[totales[p] for p in PRODUCTOS],
        marker=dict(colors=[PROD_COLORS[p] for p in PRODUCTOS],cornerradius=5),
        hovertemplate="<b>%{label}</b><extra></extra>",
        textfont=dict(size=11,family="JetBrains Mono")))
    apply_theme(fig,"Participación Anual por Producto",320)
    return fig

def f_agregacion(df, costo):
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.08,
        row_heights=[0.65,0.35],
        subplot_titles=["Producción vs Demanda (H-H)","Horas Extras & Movimientos Laborales"])
    fig.add_trace(go.Bar(x=df["Mes"],y=df["Inv_Inicial_HH"],name="Inv. Inicial",
                          marker_color="#1E3A5F",opacity=0.9),row=1,col=1)
    fig.add_trace(go.Bar(x=df["Mes"],y=df["Produccion_HH"],name="Producción H-H",
                          marker_color=C["accent"],opacity=0.9),row=1,col=1)
    fig.add_trace(go.Scatter(x=df["Mes"],y=df["Demanda_HH"],mode="lines+markers",
                              name="Demanda H-H",line=dict(color=C["a3"],dash="dash",width=2.5),
                              marker=dict(size=7)),row=1,col=1)
    fig.add_trace(go.Scatter(x=df["Mes"],y=df["Horas_Regulares"],mode="lines",name="Cap. Regular",
                              line=dict(color=C["a2"],dash="dot",width=2)),row=1,col=1)
    fig.add_trace(go.Bar(x=df["Mes"],y=df["Horas_Extras"],name="H. Extras",
                          marker_color=C["warn"],opacity=0.8),row=2,col=1)
    fig.add_trace(go.Bar(x=df["Mes"],y=df["Contratacion"],name="Contrataciones",
                          marker_color=C["ok"],opacity=0.8),row=2,col=1)
    fig.add_trace(go.Bar(x=df["Mes"],y=[-v for v in df["Despidos"]],name="Despidos",
                          marker_color=C["err"],opacity=0.8),row=2,col=1)
    fig.update_layout(**THEME, barmode="stack", height=520,
        title=dict(text=f"Plan Agregado Óptimo — COP ${costo:,.0f}",x=0.5,
                   font=dict(size=15,color=C["accent"],family="Syne, sans-serif")),
        legend=dict(bgcolor="rgba(0,0,0,0)",orientation="h",y=-0.1,x=0.5,xanchor="center"))
    for r in [1,2]:
        fig.update_xaxes(gridcolor="#1E293B",row=r,col=1)
        fig.update_yaxes(gridcolor="#1E293B",row=r,col=1)
    return fig

def f_desagregacion(desag):
    fig = make_subplots(rows=3,cols=2,vertical_spacing=0.07,horizontal_spacing=0.08,
        subplot_titles=[f"{PROD_ICONS[p]} {p.replace('_',' ')}" for p in PRODUCTOS])
    for idx, p in enumerate(PRODUCTOS):
        r, c = idx//2+1, idx%2+1
        df = desag[p]
        fig.add_trace(go.Bar(x=df["Mes"],y=df["Produccion"],name=p,
                              marker_color=PROD_COLORS[p],opacity=0.85,showlegend=False,
                              hovertemplate="%{x}<br>Prod: %{y:.0f}<extra></extra>"),row=r,col=c)
        fig.add_trace(go.Scatter(x=df["Mes"],y=df["Demanda"],mode="lines+markers",
                                  line=dict(color=C["a3"],dash="dash",width=1.5),
                                  marker=dict(size=4),showlegend=False),row=r,col=c)
        fig.add_trace(go.Scatter(x=df["Mes"],y=df["Inv_Fin"],mode="lines",name="Inv.Fin",
                                  fill="tozeroy",fillcolor=f"{PROD_COLORS[p]}15",
                                  line=dict(color=PROD_COLORS[p],width=1,dash="dot"),
                                  showlegend=False),row=r,col=c)
    fig.update_layout(**THEME,height=700,barmode="group",
        title=dict(text="Desagregación — Producción · Demanda · Inventario",x=0.5,
                   font=dict(size=15,color=C["accent"],family="Syne, sans-serif")))
    for i in range(1,4):
        for j in range(1,3):
            fig.update_xaxes(gridcolor="#1E293B",tickfont=dict(size=8),row=i,col=j)
            fig.update_yaxes(gridcolor="#1E293B",title_text="und",row=i,col=j)
    return fig

def f_gantt(df_l, n=80):
    if df_l.empty: return go.Figure()
    sub = df_l.head(n).copy().reset_index(drop=True)
    fig = go.Figure()
    for _, row in sub.iterrows():
        col = PROD_COLORS.get(row["producto"],"#aaa")
        fig.add_trace(go.Bar(x=[row["tiempo_sistema"]],y=[row["lote_id"]],
                              base=[row["t_creacion"]],orientation="h",
                              marker_color=col,opacity=0.8,showlegend=False,
                              hovertemplate=(f"<b>{row['producto']}</b><br>Lote: {row['lote_id']}<br>"
                                             f"Inicio: {row['t_creacion']:.0f} min<br>"
                                             f"Duración: {row['tiempo_sistema']:.1f} min<br>"
                                             f"Espera: {row['total_espera']:.1f} min<extra></extra>")))
    for p, col in PROD_COLORS.items():
        fig.add_trace(go.Bar(x=[None],y=[None],marker_color=col,
                              name=f"{PROD_ICONS[p]} {p.replace('_',' ')}",showlegend=True))
    apply_theme(fig,"Diagrama de Gantt — Flujo de Lotes",max(380,len(sub)*8))
    fig.update_layout(barmode="overlay",xaxis_title="Tiempo simulado (min)",yaxis_title="Lote ID",
        legend=dict(bgcolor="rgba(0,0,0,0)",orientation="h",y=-0.15,x=0.5,xanchor="center"))
    return fig

def f_colas(df_u):
    if df_u.empty: return go.Figure()
    fig = go.Figure()
    pal = list(PROD_COLORS.values())+["#FBBF24","#A78BFA"]
    for i,(rec,grp) in enumerate(df_u.groupby("recurso")):
        grp = grp.sort_values("tiempo")
        fig.add_trace(go.Scatter(x=grp["tiempo"],y=grp["cola"],mode="lines",name=rec,
                                  line=dict(color=pal[i%len(pal)],width=1.5),
                                  fill="tozeroy",fillcolor=f"{pal[i%len(pal)]}08",
                                  hovertemplate=f"<b>{rec}</b><br>t=%{{x:.0f}} min<br>Cola: %{{y}}<extra></extra>"))
    apply_theme(fig,"Evolución de Colas por Recurso (SimPy)",380)
    fig.update_xaxes(title_text="Tiempo (min)"); fig.update_yaxes(title_text="Cola")
    fig.update_layout(legend=dict(bgcolor="rgba(0,0,0,0)",orientation="h",y=-0.2,x=0.5,xanchor="center"))
    return fig

def f_utilizacion(df_u):
    df_ut = calc_utilizacion(df_u)
    if df_ut.empty: return go.Figure()
    fig = make_subplots(rows=1,cols=2,subplot_titles=["Utilización (%)","Cola Promedio"])
    cols_bar = [C["err"] if u>=80 else C["warn"] if u>=60 else C["a3"] for u in df_ut["Utilización_%"]]
    fig.add_trace(go.Bar(x=df_ut["Recurso"],y=df_ut["Utilización_%"],
                          marker_color=cols_bar,showlegend=False,
                          text=df_ut["Utilización_%"].apply(lambda v:f"{v:.1f}%"),
                          textposition="outside",
                          hovertemplate="%{x}<br><b>%{y:.2f}%</b><extra></extra>"),row=1,col=1)
    fig.add_hline(y=80,line_dash="dash",line_color=C["err"],annotation_text="⚠ 80%",row=1,col=1)
    fig.add_hline(y=60,line_dash="dot",line_color=C["warn"],annotation_text="60%",row=1,col=1)
    fig.add_trace(go.Bar(x=df_ut["Recurso"],y=df_ut["Cola Prom"],
                          marker_color=C["a2"],showlegend=False,
                          text=df_ut["Cola Prom"].apply(lambda v:f"{v:.2f}"),
                          textposition="outside",
                          hovertemplate="%{x}<br>Cola: %{y:.2f}<extra></extra>"),row=1,col=2)
    fig.update_layout(**THEME,height=380,
        title=dict(text="Utilización de Recursos & Cuellos de Botella",x=0.5,
                   font=dict(size=15,color=C["accent"],family="Syne, sans-serif")))
    fig.update_xaxes(gridcolor="#1E293B"); fig.update_yaxes(gridcolor="#1E293B")
    return fig

def f_radar(df_kpi):
    if df_kpi.empty: return go.Figure()
    cats = ["Throughput (und/h)","Cycle Time (min/und)","Lead Time (min/lote)","WIP","Cumplimiento %"]
    fig  = go.Figure()
    for _, row in df_kpi.iterrows():
        vals = [row.get(c,0) for c in cats]
        maxv = [max(df_kpi[c].max(),0.01) for c in cats]
        norm = [round(v/m*100,1) for v,m in zip(vals,maxv)]+[round(vals[0]/maxv[0]*100,1)]
        fig.add_trace(go.Scatterpolar(r=norm,theta=cats+[cats[0]],
            name=f"{PROD_ICONS.get(row['Producto'],'◈')} {row['Producto'].replace('_',' ')}",
            fill="toself",opacity=0.5,
            line=dict(color=PROD_COLORS.get(row["Producto"],"#aaa"),width=2)))
    fig.update_layout(**THEME,height=440,
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(visible=True,gridcolor="#1E293B",range=[0,100]),
                   angularaxis=dict(gridcolor="#1E293B")),
        title=dict(text="Radar de KPIs por Producto (normalizado)",x=0.5,
                   font=dict(size=15,color=C["accent"],family="Syne, sans-serif")),
        legend=dict(bgcolor="rgba(0,0,0,0)",orientation="h",y=-0.14,x=0.5,xanchor="center"))
    return fig

def f_sensores(df_s):
    if df_s.empty: return go.Figure()
    fig = make_subplots(rows=2,cols=2,shared_xaxes=False,vertical_spacing=0.12,
        subplot_titles=["🌡 Temperatura Horno (°C)","💧 Humedad Relativa (%)","⚙ Ocupación","📋 Cola"])
    # Temp
    fig.add_trace(go.Scatter(x=df_s["tiempo"],y=df_s["temperatura"],mode="lines",
                              line=dict(color="#FF6B6B",width=1.5),
                              fill="tozeroy",fillcolor="rgba(255,107,107,0.05)"),row=1,col=1)
    fig.add_hline(y=200,line_dash="dash",line_color=C["err"],annotation_text="Límite 200°C",row=1,col=1)
    fig.add_hline(y=160,line_dash="dot",line_color=C["warn"],annotation_text="Base 160°C",row=1,col=1)
    # Humedad
    if "humedad" in df_s.columns:
        fig.add_trace(go.Scatter(x=df_s["tiempo"],y=df_s["humedad"],mode="lines",
                                  line=dict(color=C["a2"],width=1.5),
                                  fill="tozeroy",fillcolor=f"{C['a2']}0A"),row=1,col=2)
    # Ocupación
    fig.add_trace(go.Scatter(x=df_s["tiempo"],y=df_s["horno_ocup"],mode="lines",
                              line=dict(color=C["a3"],width=1.5),
                              fill="tozeroy",fillcolor=f"{C['a3']}12"),row=2,col=1)
    # Cola
    fig.add_trace(go.Scatter(x=df_s["tiempo"],y=df_s["horno_cola"],mode="lines",
                              line=dict(color=C["warn"],width=1.5),
                              fill="tozeroy",fillcolor=f"{C['warn']}10"),row=2,col=2)
    fig.update_layout(**THEME,height=500,
        title=dict(text="Sensores Virtuales — Monitoreo en Tiempo Real",x=0.5,
                   font=dict(size=15,color=C["accent"],family="Syne, sans-serif")))
    for r_ in [1,2]:
        for c_ in [1,2]:
            fig.update_xaxes(gridcolor="#1E293B",title_text="Tiempo (min)",row=r_,col=c_)
            fig.update_yaxes(gridcolor="#1E293B",row=r_,col=c_)
    return fig

def f_comparacion(esc_store):
    if not esc_store: return go.Figure()
    filas = []
    for nm, v in esc_store.items():
        try:
            dk = pd.read_json(v["kpis"]) if v.get("kpis","{}")!="{}" else pd.DataFrame()
            du = pd.read_json(v["util"]) if v.get("util","{}")!="{}" else pd.DataFrame()
        except: continue
        if dk.empty: continue
        fila = {"Escenario":nm}
        for col in ["Throughput (und/h)","Lead Time (min/lote)","WIP","Cumplimiento %"]:
            if col in dk.columns: fila[col] = round(dk[col].mean(),2)
        if not du.empty and "Utilización_%" in du.columns:
            fila["Util Máx %"] = round(du["Utilización_%"].max(),2)
        filas.append(fila)
    if not filas: return go.Figure()
    df = pd.DataFrame(filas)
    metricas = [("Throughput (und/h)","🚀 Throughput"),
                ("Lead Time (min/lote)","⏱ Lead Time"),
                ("Cumplimiento %","✅ Cumplimiento"),
                ("Util Máx %","⚙ Util. Máx")]
    fig = make_subplots(rows=2,cols=2,subplot_titles=[m[1] for m in metricas])
    pal = ["#F4A261","#48CAE4","#95D5B2","#C77DFF","#FF6B6B","#FBBF24","#A78BFA","#FB7185"]
    for i,(col,_) in enumerate(metricas):
        r,c_ = i//2+1, i%2+1
        if col not in df.columns: continue
        fig.add_trace(go.Bar(x=df["Escenario"],y=df[col],
                              marker_color=[pal[j%len(pal)] for j in range(len(df))],
                              text=df[col].apply(lambda v:f"{v:.2f}"),textposition="outside",
                              showlegend=False,
                              hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:.2f}}<extra></extra>"),row=r,col=c_)
    fig.update_layout(**THEME,height=520,
        title=dict(text="Comparación de Escenarios What-If",x=0.5,
                   font=dict(size=15,color=C["accent"],family="Syne, sans-serif")))
    fig.update_xaxes(gridcolor="#1E293B",tickangle=20)
    fig.update_yaxes(gridcolor="#1E293B")
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
*{box-sizing:border-box}
body{background:#050A13!important}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:#0A1628}
::-webkit-scrollbar-thumb{background:#1E3A5F;border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:#2563EB}
.nav-btn{transition:all .2s ease!important}
.nav-btn:hover{color:#F4A261!important;background:rgba(244,162,97,.08)!important}
.nav-btn.active-nav{color:#F4A261!important;background:rgba(244,162,97,.12)!important;border-right:2px solid #F4A261!important}
.rc-slider-rail{background:#1E293B!important}
.rc-slider-track{background:linear-gradient(90deg,#2563EB,#F4A261)!important}
.rc-slider-handle{border-color:#F4A261!important;background:#F4A261!important;box-shadow:0 0 6px rgba(244,162,97,.5)!important}
.Select-control{background:#061020!important;border-color:#1E3A5F!important}
.Select-value-label{color:#E2E8F0!important}
.Select-menu-outer{background:#0A1628!important;border-color:#1E3A5F!important}
.VirtualizedSelectOption{color:#94A3B8!important}
.VirtualizedSelectFocusedOption{background:#1E3A5F!important;color:#E2E8F0!important}
input[type=checkbox]{accent-color:#F4A261}
.form-check-input:checked{background-color:#F4A261!important;border-color:#F4A261!important}
.form-check-input[type="checkbox"][role="switch"]:checked{background-color:#F4A261!important}
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
NAV_ITEMS = [
    ("01","DEMANDA",       "📊","tab-demanda"),
    ("02","PLANEACIÓN",    "📈","tab-agregacion"),
    ("03","DESAGREGACIÓN", "📉","tab-desag"),
    ("04","SIMULACIÓN",    "⚙️","tab-sim"),
    ("05","KPIs",          "🎯","tab-kpis"),
    ("06","SENSORES",      "🌡","tab-sensores"),
    ("07","ESCENARIOS",    "🔀","tab-escenarios"),
]
TABS_SIM   = {"tab-sim","tab-kpis","tab-sensores"}
TABS_MES   = {"tab-desag"}
TABS_NOPAR = {"tab-demanda","tab-agregacion","tab-escenarios"}

app = dash.Dash(__name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap",
        dbc.themes.CYBORG,
    ],
    suppress_callback_exceptions=True,
    title="Gemelo Digital — Dora del Hoyo")

def nav_btn(num, label, icon, tid):
    return html.Button([
        html.Div([
            html.Span(num,style={"fontSize":"8px","color":C["muted"],"display":"block",
                                  "fontFamily":"JetBrains Mono","lineHeight":"1"}),
            html.Span(icon,style={"fontSize":"15px","display":"block","lineHeight":"1.3"}),
        ], style={"width":"30px","flexShrink":"0","textAlign":"center"}),
        html.Span(label,style={"fontSize":"11px","letterSpacing":"0.1em",
                                "fontFamily":"Syne, sans-serif","fontWeight":"700"}),
    ], id=f"btn-{tid}", n_clicks=0, className="nav-btn",
    style={"background":"transparent","border":"none","color":C["muted"],
           "width":"100%","textAlign":"left","padding":"10px 12px",
           "cursor":"pointer","display":"flex","alignItems":"center","gap":"10px"})

sidebar = html.Div([
    html.Div([
        html.Div("🥐",style={"fontSize":"30px","lineHeight":"1","marginBottom":"4px"}),
        html.Div("DORA DEL HOYO",style={"fontSize":"10px","fontWeight":"800","letterSpacing":"0.2em",
                                         "color":C["text"],"fontFamily":"Syne, sans-serif"}),
        html.Div("GEMELO DIGITAL v2",style={"fontSize":"8px","color":C["accent"],"letterSpacing":"0.2em",
                                              "fontFamily":"JetBrains Mono","marginTop":"2px"}),
        html.Div(style={"height":"1px","background":f"linear-gradient(90deg,{C['accent']},transparent)","marginTop":"10px"}),
    ], style={"padding":"20px 14px 14px","marginBottom":"8px"}),
    html.Div([nav_btn(n,l,ic,t) for n,l,ic,t in NAV_ITEMS]),
    html.Div([
        html.Div(style={"height":"1px","background":f"linear-gradient(90deg,{C['border']},transparent)","marginBottom":"8px"}),
        html.Div(id="sidebar-status",style={"fontSize":"9px","color":C["muted"],"fontFamily":"JetBrains Mono"}),
    ], style={"position":"absolute","bottom":"16px","left":"0","width":"100%","padding":"0 14px"}),
], style={"width":"185px","minHeight":"100vh","background":C["bg"],"borderRight":f"1px solid {C['border']}",
          "display":"flex","flexDirection":"column","position":"fixed","top":"0","left":"0","zIndex":"200",
          "boxShadow":"4px 0 24px rgba(0,0,0,.5)"})

stores = html.Div([
    dcc.Store(id="s-tab",     data="tab-demanda"),
    dcc.Store(id="s-agr",     data=None),
    dcc.Store(id="s-desag",   data=None),
    dcc.Store(id="s-sim",     data=None),
    dcc.Store(id="s-util",    data=None),
    dcc.Store(id="s-kpis",    data=None),
    dcc.Store(id="s-sen",     data=None),
    dcc.Store(id="s-plan",    data=None),
    dcc.Store(id="s-esc",     data={}),
    dcc.Store(id="p-mes",     data=1),
    dcc.Store(id="p-factor",  data=1.0),
    dcc.Store(id="p-horno",   data=3),
    dcc.Store(id="p-opciones",data=[]),
])

app.layout = html.Div([
    html.Div(CSS, dangerouslyAllowHTML=True),
    stores, sidebar,
    html.Div([
        # Header
        html.Div([
            html.Div([
                html.Div([html.Span("●",style={"color":C["ok"],"marginRight":"5px","fontSize":"8px"}),
                          html.Span("SISTEMA EN LÍNEA",style={"fontFamily":"JetBrains Mono","fontSize":"9px",
                                                               "letterSpacing":"0.18em","color":C["muted"]})],
                         style={"display":"flex","alignItems":"center","marginBottom":"4px"}),
                html.Div(id="h-name",children="DEMANDA",style={
                    "fontFamily":"Syne, sans-serif","fontWeight":"800",
                    "fontSize":"24px","color":C["text"],"letterSpacing":"0.04em"}),
            ]),
            html.Div([
                html.Div([
                    html.Div(f"{sum(sum(DEM_HISTORICA[p]) for p in PRODUCTOS):,}",
                             style={"fontSize":"16px","fontWeight":"800","color":C["accent"],"fontFamily":"Syne"}),
                    html.Div("und/año",style={"fontSize":"9px","color":C["muted"],"fontFamily":"JetBrains Mono"}),
                ], style={"textAlign":"center","padding":"0 14px","borderLeft":f"1px solid {C['border']}"}),
                html.Div([
                    html.Div("5",style={"fontSize":"16px","fontWeight":"800","color":C["a2"],"fontFamily":"Syne"}),
                    html.Div("productos",style={"fontSize":"9px","color":C["muted"],"fontFamily":"JetBrains Mono"}),
                ], style={"textAlign":"center","padding":"0 14px","borderLeft":f"1px solid {C['border']}"}),
            ], style={"display":"flex","alignItems":"center"}),
        ], style={"padding":"16px 28px","display":"flex","justifyContent":"space-between",
                   "alignItems":"center","background":C["bg"],"borderBottom":f"1px solid {C['border']}"}),
        # Config
        html.Div(id="cfg-wrapper",
                 style={**CARD,"borderRadius":"0","borderLeft":"none","borderRight":"none","borderTop":"none"}),
        # Content
        html.Div([dcc.Loading(type="dot",color=C["accent"],
                               children=html.Div(id="tab-content"))],
                 style={"padding":"24px 28px"}),
    ], style={"marginLeft":"185px","minHeight":"100vh","background":C["bg"]}),
])

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG POR TAB
# ─────────────────────────────────────────────────────────────────────────────
def build_cfg(tab):
    btn = run_btn(); st = status_div()
    if tab in TABS_NOPAR:
        return html.Div([
            html.Span("SIN PARÁMETROS ADICIONALES PARA ESTA SECCIÓN",style={**LABEL,"marginBottom":"12px"}),
            dbc.Row([dbc.Col(btn,width="auto"),dbc.Col(st,width=True)],align="center"),
        ])
    if tab in TABS_MES:
        return html.Div([
            html.Span("MES A DESTACAR",style=LABEL),
            dbc.Row([
                dbc.Col([dcc.Dropdown(id="dd-mes",
                    options=[{"label":f"📅 {m}","value":i} for i,m in enumerate(MESES)],
                    value=1,clearable=False,style=INP,className="custom-dropdown")],width=3),
                dbc.Col(btn,width="auto"),dbc.Col(st,width=True),
            ],align="center",className="g-2"),
        ])
    # Sim full
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Span("MES A SIMULAR",style=LABEL),
                dcc.Dropdown(id="dd-mes",
                    options=[{"label":f"📅 {m}","value":i} for i,m in enumerate(MESES)],
                    value=1,clearable=False,style=INP,className="custom-dropdown"),
            ],width=3),
            dbc.Col([pslider("FACTOR DEMANDA","sl-demanda",0.5,2.0,0.1,1.0,
                              {0.5:"0.5×",1.0:"1×",1.5:"1.5×",2.0:"2×"})],width=3),
            dbc.Col([pslider("CAPACIDAD HORNO","sl-horno",1,6,1,3,
                              {i:str(i) for i in range(1,7)})],width=3),
            dbc.Col([
                html.Span("OPCIONES",style=LABEL),
                dbc.Checklist(id="chk-opciones",
                    options=[{"label":"⚠ Falla en Horno","value":"falla"},
                              {"label":"🌙 Doble Turno −20%","value":"turno"}],
                    value=[],switch=True,
                    style={"color":C["text"],"fontSize":"12px","fontFamily":"JetBrains Mono"}),
            ],width=3),
        ],className="g-3",align="center"),
        html.Div(style={"height":"8px"}),
        dbc.Row([dbc.Col(btn,width="auto"),dbc.Col(st,width=True)],align="center"),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(Output("s-tab","data"),Output("h-name","children"),
              [Input(f"btn-{t}","n_clicks") for _,l,_ic,t in NAV_ITEMS],
              prevent_initial_call=True)
def nav_click(*_):
    ctx = callback_context
    if not ctx.triggered: return dash.no_update, dash.no_update
    tid = ctx.triggered[0]["prop_id"].split(".")[0].replace("btn-","")
    lab = next(l for _,l,_ic,t in NAV_ITEMS if t==tid)
    return tid, lab

@app.callback(Output("cfg-wrapper","children"),Input("s-tab","data"))
def render_cfg(tab): return build_cfg(tab or "tab-demanda")

@app.callback(
    Output("p-mes","data"),Output("p-factor","data"),
    Output("p-horno","data"),Output("p-opciones","data"),
    Input("dd-mes","value"),Input("sl-demanda","value"),
    Input("sl-horno","value"),Input("chk-opciones","value"),
    State("p-mes","data"),State("p-factor","data"),
    State("p-horno","data"),State("p-opciones","data"),
    prevent_initial_call=True)
def sync(mes,factor,horno,opciones,sm,sf,sh,so):
    return (mes if mes is not None else sm, factor if factor is not None else sf,
            horno if horno is not None else sh, opciones if opciones is not None else so)

@app.callback(
    Output("s-agr","data"),Output("s-desag","data"),Output("s-sim","data"),
    Output("s-util","data"),Output("s-kpis","data"),Output("s-sen","data"),
    Output("s-plan","data"),Output("run-status","children"),
    Input("btn-run","n_clicks"),
    State("p-mes","data"),State("p-factor","data"),
    State("p-horno","data"),State("p-opciones","data"),
    prevent_initial_call=True)
def pipeline(n, mes_idx, factor, horno, opciones):
    if not n: return (None,)*7+("",)
    mes_idx = mes_idx if mes_idx is not None else 1
    factor  = factor  if factor  is not None else 1.0
    horno   = horno   if horno   is not None else 3
    opciones= opciones or []
    try:
        dem_h = _dem_horas(factor)
        df_agr, costo = run_agregacion(dem_h)
        prod_hh = dict(zip(MESES_FULL, df_agr["Produccion_HH"]))
        desag   = run_desagregacion(prod_hh, factor)
        mes_nm  = MESES_FULL[mes_idx]
        mes_s   = MESES[mes_idx]
        plan    = {p:int(desag[p].loc[desag[p]["Mes"]==mes_s,"Produccion"].values[0])
                   for p in PRODUCTOS}
        cap_rec = {**CAPACIDAD_BASE,"horno":int(horno)}
        ft      = 0.80 if "turno" in opciones else 1.0
        falla   = "falla" in opciones
        df_l, df_u, df_s = run_simulacion(plan, cap_rec, falla, ft)
        df_kpi = calc_kpis(df_l, plan)
        status = [html.Span("✓ ",style={"color":C["ok"],"fontWeight":"bold"}),
                  html.Span(f"Pipeline completado · {len(df_l)} lotes · {mes_nm} · COP ${costo:,.0f}",
                            style={"color":C["muted"]})]
        return (df_agr.to_json(),
                {p:df.to_json() for p,df in desag.items()},
                df_l.to_json() if not df_l.empty else "{}",
                df_u.to_json() if not df_u.empty else "{}",
                df_kpi.to_json() if not df_kpi.empty else "{}",
                df_s.to_json() if not df_s.empty else "{}",
                plan, status)
    except Exception as e:
        return (None,)*7+([html.Span("✗ ",style={"color":C["err"]}),
                            html.Span(str(e)[:200],style={"color":C["muted"]})])

@app.callback(Output("s-esc","data"),
              Input("btn-esc","n_clicks"),
              State("dd-esc","value"),State("s-plan","data"),State("s-esc","data"),
              prevent_initial_call=True)
def run_esc(n, sel, plan, esc_store):
    if not n or not plan or not sel: return esc_store or {}
    ESC = {
        "base":        {"fd":1.0,"falla":False,"ft":1.0,"dh":0},
        "demanda_20":  {"fd":1.2,"falla":False,"ft":1.0,"dh":0},
        "demanda_50":  {"fd":1.5,"falla":False,"ft":1.0,"dh":0},
        "falla_horno": {"fd":1.0,"falla":True, "ft":1.0,"dh":0},
        "red_cap":     {"fd":1.0,"falla":False,"ft":1.0,"dh":-1},
        "amp_cap":     {"fd":1.0,"falla":False,"ft":1.0,"dh":2},
        "doble_turno": {"fd":1.0,"falla":False,"ft":0.80,"dh":0},
        "optimizado":  {"fd":1.0,"falla":False,"ft":0.85,"dh":1},
    }
    res = dict(esc_store or {})
    for nm in sel:
        cfg = ESC.get(nm, ESC["base"])
        plan_aj = {p:max(int(u*cfg["fd"]),0) for p,u in plan.items()}
        cap_r   = {**CAPACIDAD_BASE,"horno":max(CAPACIDAD_BASE["horno"]+cfg.get("dh",0),1)}
        df_l,df_u,_ = run_simulacion(plan_aj,cap_r,cfg["falla"],cfg["ft"])
        dk = calc_kpis(df_l,plan_aj); du = calc_utilizacion(df_u)
        res[nm] = {"kpis":dk.to_json() if not dk.empty else "{}",
                   "util":du.to_json() if not du.empty else "{}"}
    return res

@app.callback(Output("fig-comp","figure"),Input("s-esc","data"),prevent_initial_call=True)
def upd_comp(esc): return f_comparacion(esc or {})

# ─────────────────────────────────────────────────────────────────────────────
# RENDER TAB
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(Output("tab-content","children"),
    Input("s-tab","data"),
    State("s-agr","data"),State("s-desag","data"),State("s-sim","data"),
    State("s-util","data"),State("s-kpis","data"),State("s-sen","data"),
    State("s-plan","data"),State("s-esc","data"),State("p-mes","data"))
def render(tab,agr_j,desag_j,sim_j,util_j,kpi_j,sen_j,plan,esc_store,mes_idx):
    tab     = tab or "tab-demanda"
    mes_idx = mes_idx if mes_idx is not None else 1

    def no_data():
        return html.Div([
            html.Div("🏭",style={"fontSize":"56px","textAlign":"center","padding":"40px 0 8px"}),
            html.Div("Ejecuta el pipeline primero",style={"textAlign":"center","color":C["muted"],
                      "fontSize":"16px","fontFamily":"Syne, sans-serif","fontWeight":"700"}),
            html.Div("Configura parámetros y presiona ▶ EJECUTAR PIPELINE",
                     style={"textAlign":"center","color":C["dim"],"fontSize":"11px",
                            "fontFamily":"JetBrains Mono","marginTop":"8px"}),
        ])

    cfg = {"displayModeBar":False,"responsive":True}

    # ══ DEMANDA ════════════════════════════════════════════════════════════════
    if tab == "tab-demanda":
        total_a = sum(sum(DEM_HISTORICA[p]) for p in PRODUCTOS)
        pico    = max(MESES, key=lambda m:sum(DEM_HISTORICA[p][MESES.index(m)] for p in PRODUCTOS))
        lider   = max(PRODUCTOS, key=lambda p:sum(DEM_HISTORICA[p]))
        return html.Div([
            sec("Análisis de Demanda Histórica",
                "Estacionalidad · Participación · Tendencia · 5 productos",badge="12 MESES"),
            html.Div([
                kpi("Total Anual",f"{total_a:,}","und",C["accent"],"📦","Suma todos los productos"),
                kpi("Mes Pico",pico,"",C["a2"],"📅","Mayor demanda histórica"),
                kpi("Producto Líder",lider.replace("_"," "),f"({sum(DEM_HISTORICA[lider]):,} und)",C["a3"],"🏆","Líder en volumen anual"),
                kpi("SKUs","5","productos",C["warn"],"🏷","Líneas artesanales activas"),
            ], style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"24px"}),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=f_barras(),config=cfg),width=8),
                dbc.Col(dcc.Graph(figure=f_heatmap(),config=cfg),width=4),
            ], className="g-3 mb-4"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=f_tendencia(),config=cfg),width=6),
                dbc.Col(dcc.Graph(figure=f_treemap(),config=cfg),width=6),
            ], className="g-3 mb-4"),
            sec("Tabla de Demanda Histórica","Unidades por mes y producto"),
            dtable(pd.DataFrame(DEM_HISTORICA,index=MESES)
                     .reset_index().rename(columns={"index":"Mes"}),"tbl-dem"),
        ])

    # ══ PLANEACIÓN AGREGADA ════════════════════════════════════════════════════
    if tab == "tab-agregacion":
        if not agr_j: return no_data()
        df_agr = pd.read_json(agr_j)
        costo  = df_agr["Produccion_HH"].sum()*PARAMS_AGRE["Ct"]
        bl_max = df_agr["Backlog_HH"].max()
        he_max = df_agr["Horas_Extras"].max()
        return html.Div([
            sec("Planeación Agregada Óptima","Optimización lineal PuLP/CBC — 12 períodos",badge="ÓPTIMO"),
            html.Div([
                kpi("Producción Total",f"{df_agr['Produccion_HH'].sum():,.0f}","H-H",C["accent"],"⚡","Horizonte 12 meses"),
                kpi("Costo Óptimo",f"${costo/1e6:.2f}M","COP",C["a2"],"💰","Función objetivo"),
                kpi("Backlog Acumulado",f"{df_agr['Backlog_HH'].sum():,.1f}","H-H",
                    C["err"] if df_agr["Backlog_HH"].sum()>0 else C["ok"],"⚠","Demanda diferida"),
                kpi("Horas Extra Total",f"{df_agr['Horas_Extras'].sum():,.1f}","H-H",
                    C["warn"] if df_agr["Horas_Extras"].sum()>0 else C["ok"],"🕐","Costo adicional"),
                kpi("Balance Laboral",f"{df_agr['Contratacion'].sum()-df_agr['Despidos'].sum():,.0f}","",
                    C["a3"],"👷","Contrat. − Despidos"),
            ], style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"24px"}),
            dcc.Graph(figure=f_agregacion(df_agr,costo),config=cfg),
            html.Div(style={"height":"24px"}),
            sec("Tabla del Plan Agregado","Detalle mensual en H-H"),
            dtable(df_agr,"tbl-agr"),
            html.Div(style={"height":"20px"}),
            sec("Alertas Automáticas","Análisis de riesgo operativo"),
            alert(f"Backlog máximo: {df_agr.loc[df_agr['Backlog_HH'].idxmax(),'Mes']} — {bl_max:.1f} H-H","warn")
              if bl_max > 0 else alert("Sin backlog en ningún período ✓","ok"),
            alert(f"Horas extras máx: {df_agr.loc[df_agr['Horas_Extras'].idxmax(),'Mes']} — {he_max:.1f} H-H","info" if True else "ok")
              if he_max > 0 else alert("Sin horas extras requeridas ✓","ok"),
            alert(f"Meses con inventario negativo: {(df_agr['Inv_Final_HH']<0).sum()}","err")
              if (df_agr["Inv_Final_HH"]<0).sum()>0 else alert("Inventario siempre positivo ✓","ok"),
        ])

    # ══ DESAGREGACIÓN ══════════════════════════════════════════════════════════
    if tab == "tab-desag":
        if not desag_j: return no_data()
        desag_d = {p:pd.read_json(v) for p,v in desag_j.items()}
        mes_lbl = MESES[mes_idx]
        tot_und = sum(desag_d[p]["Produccion"].sum() for p in PRODUCTOS)
        return html.Div([
            sec("Desagregación del Plan","Distribución óptima por producto y mes",badge=f"FOCO: {mes_lbl}"),
            html.Div([
                kpi("Total Unidades",f"{tot_und:,.0f}","und",C["accent"],"📦","12 meses · todos productos"),
                *[kpi(f"{PROD_ICONS[p]} {p.replace('_',' ').replace('Mantecadas ','Mant. ')}",
                      f"{desag_d[p]['Produccion'].sum():,.0f}","und",PROD_COLORS[p],"",
                      f"Backlog: {desag_d[p]['Backlog'].sum():.0f} und") for p in PRODUCTOS],
            ], style={"display":"flex","gap":"10px","flexWrap":"wrap","marginBottom":"24px"}),
            dcc.Graph(figure=f_desagregacion(desag_d),config=cfg),
            html.Div(style={"height":"24px"}),
            sec(f"Detalle por Producto — Mes Foco: {mes_lbl}","Producción · Inventario · Backlog"),
            dbc.Row([
                dbc.Col([
                    html.Div(f"{PROD_ICONS[p]} {p.replace('_',' ')}",
                             style={"fontFamily":"Syne","fontWeight":"800","color":PROD_COLORS[p],
                                    "marginBottom":"8px","fontSize":"13px"}),
                    dtable(desag_d[p],f"tbl-d-{i}",ps=6),
                ], width=6, className="mb-3")
                for i,p in enumerate(PRODUCTOS)
            ]),
        ])

    # ══ SIMULACIÓN ═════════════════════════════════════════════════════════════
    if tab == "tab-sim":
        if not sim_j or sim_j=="{}": return no_data()
        df_l = pd.read_json(sim_j)
        if df_l.empty: return no_data()
        n_l   = len(df_l)
        t_tot = round(df_l["t_fin"].max(),1)
        d_prm = round(df_l["tiempo_sistema"].mean(),1)
        e_prm = round(df_l["total_espera"].mean(),1)
        pct_e = round(e_prm/max(d_prm,0.01)*100,1)
        df_u  = pd.read_json(util_j) if util_j and util_j!="{}" else pd.DataFrame()
        return html.Div([
            sec("Simulación de Eventos Discretos","Motor SimPy · Rutas por etapas · Recursos compartidos",badge="SIMPY"),
            html.Div([
                kpi("Lotes Simulados",n_l,"lotes",C["accent"],"🔄","Total período"),
                kpi("Tiempo Total",f"{t_tot:,.0f}","min",C["a2"],"⏱",f"{t_tot/60:.1f} horas"),
                kpi("Lead Time Prom",f"{d_prm:,.1f}","min/lote",C["a3"],"🏁","Entrada→Salida"),
                kpi("Espera Prom",f"{e_prm:,.1f}","min",C["warn"],"⏳",f"{pct_e:.0f}% del LT es espera"),
                kpi("Productos",len(df_l["producto"].unique()),"SKUs",C["err"],"🏷","En simulación"),
            ], style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"24px"}),
            dcc.Graph(figure=f_gantt(df_l,80),config=cfg),
            html.Div(style={"height":"20px"}),
            dcc.Graph(figure=f_colas(df_u),config=cfg),
            html.Div(style={"height":"24px"}),
            sec("Registro de Lotes","Primeros 200 lotes completados"),
            dtable(df_l.head(200)[["lote_id","producto","tamano","t_creacion",
                                    "t_fin","tiempo_sistema","total_espera"]],"tbl-lotes"),
        ])

    # ══ KPIs ═══════════════════════════════════════════════════════════════════
    if tab == "tab-kpis":
        if not kpi_j or kpi_j=="{}": return no_data()
        df_kpi = pd.read_json(kpi_j)
        df_u   = pd.read_json(util_j) if util_j and util_j!="{}" else pd.DataFrame()
        df_utc = calc_utilizacion(df_u)
        cb     = df_utc.iloc[0]["Recurso"]      if not df_utc.empty else "N/A"
        cb_u   = df_utc.iloc[0]["Utilización_%"] if not df_utc.empty else 0
        return html.Div([
            sec("KPIs Operativos & Cuellos de Botella","Throughput · Lead Time · WIP · Takt · Cumplimiento",badge="KPIs"),
            html.Div([
                html.Span("🚨 CUELLO DE BOTELLA: ",style={"fontFamily":"Syne","fontWeight":"800",
                           "fontSize":"13px","color":C["err"] if cb_u>=80 else C["warn"]}),
                html.Span(f"{cb.upper()}  ·  {cb_u:.1f}% utilización",
                          style={"fontFamily":"JetBrains Mono","fontSize":"12px","color":C["text"]}),
            ], style={"background":f"{C['err']}12","border":f"1px solid {C['err']}30",
                       "borderLeft":f"4px solid {C['err']}","borderRadius":"8px",
                       "padding":"12px 18px","marginBottom":"24px"})
              if cb_u >= 80 else alert(f"✓ Sin cuellos de botella críticos — mayor utilización: {cb} ({cb_u:.1f}%)","ok"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=f_radar(df_kpi),config=cfg),width=6),
                dbc.Col(dcc.Graph(figure=f_utilizacion(df_u),config=cfg),width=6),
            ],className="g-3 mb-4"),
            sec("Tabla KPIs por Producto"),
            dtable(df_kpi,"tbl-kpis",ps=10),
            html.Div(style={"height":"20px"}),
            sec("Utilización de Recursos"),
            dtable(df_utc,"tbl-util",ps=10),
        ])

    # ══ SENSORES ════════════════════════════════════════════════════════════════
    if tab == "tab-sensores":
        if not sen_j or sen_j=="{}": return no_data()
        df_s = pd.read_json(sen_j)
        if df_s.empty: return no_data()
        t_max  = round(df_s["temperatura"].max(),1)
        t_min  = round(df_s["temperatura"].min(),1)
        t_prom = round(df_s["temperatura"].mean(),1)
        t_std  = round(df_s["temperatura"].std(),1)
        sobr   = int((df_s["temperatura"]>195).sum())
        return html.Div([
            sec("Sensores Virtuales — Monitor de Planta","Temperatura · Humedad · Ocupación · Cola del Horno",badge="LIVE"),
            html.Div([
                kpi("Temp. Máxima",t_max,"°C",C["err"] if t_max>=195 else C["warn"],"🌡",
                    "⚠ ALARMA" if t_max>=195 else "Dentro del rango"),
                kpi("Temp. Mínima",t_min,"°C",C["a2"],"❄","Base teórica: 160°C"),
                kpi("Temp. Promedio",t_prom,"°C",C["accent"],"📊",f"σ = {t_std}°C"),
                kpi("Sobrecalentamientos",sobr,"eventos",C["err"] if sobr>0 else C["ok"],"🔥",
                    "Lecturas > 195°C"),
                kpi("Total Lecturas",len(df_s),"muestras",C["a3"],"📡","Intervalo: 10 min"),
            ], style={"display":"flex","gap":"12px","flexWrap":"wrap","marginBottom":"24px"}),
            dcc.Graph(figure=f_sensores(df_s),config=cfg),
            html.Div(style={"height":"16px"}),
            alert(f"⚠ {sobr} lecturas con temperatura > 195°C. Revisar sistema de refrigeración.","err")
              if sobr>0 else alert("✓ Temperatura dentro de parámetros normales durante toda la simulación.","ok"),
        ])

    # ══ ESCENARIOS ══════════════════════════════════════════════════════════════
    if tab == "tab-escenarios":
        esc_opts = [
            {"label":"⚙ Base (sin cambios)","value":"base"},
            {"label":"📈 Demanda +20%","value":"demanda_20"},
            {"label":"📈 Demanda +50%","value":"demanda_50"},
            {"label":"🔥 Falla de Horno","value":"falla_horno"},
            {"label":"➖ Reducir Capacidad Horno","value":"red_cap"},
            {"label":"➕ Ampliar Capacidad Horno (+2)","value":"amp_cap"},
            {"label":"🌙 Doble Turno (−20% tiempos)","value":"doble_turno"},
            {"label":"🚀 Optimizado","value":"optimizado"},
        ]
        has = bool(esc_store)
        # Tabla resumen
        tbl_esc = None
        if has:
            rows = []
            for nm, v in esc_store.items():
                try:
                    dk = pd.read_json(v["kpis"]) if v.get("kpis","{}")!="{}" else pd.DataFrame()
                    du = pd.read_json(v["util"]) if v.get("util","{}")!="{}" else pd.DataFrame()
                    if dk.empty: continue
                    rows.append({
                        "Escenario":nm,
                        "Throughput (und/h)": round(dk["Throughput (und/h)"].mean(),2) if "Throughput (und/h)" in dk else 0,
                        "Lead Time (min)":    round(dk["Lead Time (min/lote)"].mean(),2) if "Lead Time (min/lote)" in dk else 0,
                        "Cumplimiento %":     round(dk["Cumplimiento %"].mean(),2) if "Cumplimiento %" in dk else 0,
                        "WIP":                round(dk["WIP"].mean(),2) if "WIP" in dk else 0,
                        "Util Máx %":         round(du["Utilización_%"].max(),2) if not du.empty and "Utilización_%" in du else 0,
                    })
                except: pass
            if rows: tbl_esc = dtable(pd.DataFrame(rows),"tbl-esc")

        return html.Div([
            sec("Análisis de Escenarios What-If","Evaluación comparativa de estrategias operativas",badge="WHAT-IF"),
            html.Div([
                html.Span("SELECCIONAR ESCENARIOS",style={**LABEL,"marginBottom":"12px"}),
                dcc.Checklist(id="dd-esc",options=esc_opts,value=["base","demanda_20","falla_horno"],
                    inline=True,labelStyle={"marginRight":"18px","display":"inline-flex",
                                            "alignItems":"center","gap":"5px","cursor":"pointer"},
                    style={"color":C["text"],"fontSize":"12px","fontFamily":"JetBrains Mono","lineHeight":"2.2"}),
                html.Div(style={"height":"14px"}),
                html.Button([html.Span("▶ "),"CORRER ESCENARIOS"],id="btn-esc",n_clicks=0,
                    style={"background":"transparent","color":C["a2"],"border":f"1px solid {C['a2']}",
                           "padding":"10px 24px","fontFamily":"Syne, sans-serif","fontWeight":"700",
                           "fontSize":"13px","letterSpacing":"0.08em","borderRadius":"8px","cursor":"pointer",
                           "boxShadow":f"0 0 12px {C['a2']}20"}),
            ], style={**CARD,"marginBottom":"24px"}),
            dcc.Graph(id="fig-comp",figure=f_comparacion(esc_store),config=cfg),
            html.Div(style={"height":"20px"}),
            alert("Selecciona escenarios y pulsa ▶ CORRER para compararlos. 'Base' es el punto de referencia.","info")
              if not has else None,
            html.Div([sec("Resumen Numérico de Escenarios"),tbl_esc]) if tbl_esc else None,
        ])

    return html.Div("Selecciona una sección en el menú.",style={"color":C["muted"],"padding":"40px"})

# ─────────────────────────────────────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────────────────────────────────────
def _run(): app.run(debug=False,port=8050,host="0.0.0.0",use_reloader=False)

print("\n"+"═"*60)
print("  🥐 GEMELO DIGITAL — PANADERÍA DORA DEL HOYO v2.0")
print("  ▶  http://127.0.0.1:8050")
print("═"*60+"\n")

threading.Thread(target=_run, daemon=True).start()
try:
    from google.colab.output import eval_js
    url = eval_js("google.colab.kernel.proxyPort(8050)")
    print(f"✅ Colab → {url}")
    from IPython.display import IFrame, display
    display(IFrame(src=url,width="100%",height="900px"))
except Exception:
    import time; time.sleep(2)
    print("▶ Abrir http://127.0.0.1:8050 en tu navegador")

