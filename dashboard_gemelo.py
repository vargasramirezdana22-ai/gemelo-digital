"""
dashboard_gemelo.py
===================
Dashboard profesional Dash — Gemelo Digital Dora del Hoyo
==========================================================

Integra en una sola app web los módulos:
  demanda.py / agregacion.py / desagregacion_hh.py / gemelo_digital.py

INSTALACIÓN:
    pip install dash dash-bootstrap-components simpy pulp pandas numpy plotly statsmodels

EJECUCIÓN:
    python dashboard_gemelo.py
    → Abrir http://127.0.0.1:8050 en el navegador

    En Google Colab:
        from pyngrok import ngrok
        !python dashboard_gemelo.py &
        public_url = ngrok.connect(8050)
        print(public_url)
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import math
import random
import warnings
import numpy as np
import pandas as pd
import simpy
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pulp import (
    LpProblem, LpMinimize, LpVariable, LpStatus, lpSum, value, PULP_CBC_CMD
)
import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# DATOS MAESTROS  (espejo de gemelo_digital.py — autocontenido)
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

PRODUCTOS = [
    "Brownies", "Mantecadas", "Mantecadas_Amapola",
    "Torta_Naranja", "Pan_Maiz",
]
MESES = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December",
]
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
INV_INICIAL = {p: 0 for p in PRODUCTOS}

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
TAMANO_LOTE_BASE = {
    "Brownies":12,"Mantecadas":10,"Mantecadas_Amapola":10,
    "Torta_Naranja":12,"Pan_Maiz":15,
}
CAPACIDAD_BASE = {
    "mezcla":2,"dosificado":2,"horno":3,
    "enfriamiento":4,"empaque":2,"amasado":1,
}
PARAMS_AGRE = {
    "Ct":4310,"Ht":100000,"PIt":100000,"CRt":11364,"COt":14205,
    "CW_mas":14204,"CW_menos":15061,"M":1,"LR_inicial":44*4*10,"inv_seg":0.0,
}

# Paleta de colores por producto
PROD_COLORS = {
    "Brownies":"#E8A838","Mantecadas":"#4FC3F7","Mantecadas_Amapola":"#81C784",
    "Torta_Naranja":"#CE93D8","Pan_Maiz":"#FF8A65",
}

# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# MOTOR DE CÁLCULO
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

def _dem_horas(factores=None):
    if factores is None:
        factores = {m: 1.0 for m in MESES}
    return {
        mes: round(
            sum(DEM_HISTORICA[p][i]*HORAS_PRODUCTO[p] for p in PRODUCTOS) * factores.get(mes,1.0),
            4
        )
        for i, mes in enumerate(MESES)
    }

def run_agregacion(dem_horas, params=None):
    if params is None:
        params = PARAMS_AGRE.copy()
    Ct,Ht,PIt = params["Ct"],params["Ht"],params["PIt"]
    CRt,COt   = params["CRt"],params["COt"]
    Wm,Wd     = params["CW_mas"],params["CW_menos"]
    M,LRi     = params["M"],params["LR_inicial"]
    meses     = MESES

    mdl = LpProblem("Agr",LpMinimize)
    P  = LpVariable.dicts("P", meses, lowBound=0)
    I  = LpVariable.dicts("I", meses, lowBound=0)
    S  = LpVariable.dicts("S", meses, lowBound=0)
    LR = LpVariable.dicts("LR",meses, lowBound=0)
    LO = LpVariable.dicts("LO",meses, lowBound=0)
    LU = LpVariable.dicts("LU",meses, lowBound=0)
    NI = LpVariable.dicts("NI",meses)
    Wmas  = LpVariable.dicts("Wm",meses, lowBound=0)
    Wmenos= LpVariable.dicts("Wd",meses, lowBound=0)

    mdl += lpSum(Ct*P[t]+Ht*I[t]+PIt*S[t]+CRt*LR[t]+COt*LO[t]+Wm*Wmas[t]+Wd*Wmenos[t]
                 for t in meses), "FO"
    for idx,t in enumerate(meses):
        d = dem_horas[t]; tp = meses[idx-1] if idx>0 else None
        if idx==0: mdl += NI[t]==0+P[t]-d, f"NI0_{t}"
        else:       mdl += NI[t]==NI[tp]+P[t]-d, f"NI_{t}"
        mdl += NI[t]==I[t]-S[t],              f"NId_{t}"
        mdl += LU[t]+LO[t]==M*P[t],            f"Hb_{t}"
        mdl += LU[t]<=LR[t],                   f"LUl_{t}"
        if idx==0: mdl += LR[t]==LRi+Wmas[t]-Wmenos[t], f"LR0_{t}"
        else:       mdl += LR[t]==LR[tp]+Wmas[t]-Wmenos[t], f"LR_{t}"

    mdl.solve(PULP_CBC_CMD(msg=False))
    costo = value(mdl.objective)

    ini_l, fin_l = [], []
    for idx,t in enumerate(meses):
        ini = 0.0 if idx==0 else fin_l[-1]
        ini_l.append(ini)
        fin_l.append(ini+P[t].varValue-dem_horas[t])

    df = pd.DataFrame({
        "Mes":meses,
        "Demanda_HH":[round(dem_horas[t],2) for t in meses],
        "Produccion_HH":[round(P[t].varValue,2) for t in meses],
        "Backlog_HH":[round(S[t].varValue,2) for t in meses],
        "Horas_Regulares":[round(LR[t].varValue,2) for t in meses],
        "Horas_Extras":[round(LO[t].varValue,2) for t in meses],
        "Inventario_Inicial_HH":[round(v,2) for v in ini_l],
        "Inventario_Final_HH":[round(v,2) for v in fin_l],
        "Contratacion":[round(Wmas[t].varValue,2) for t in meses],
        "Despidos":[round(Wmenos[t].varValue,2) for t in meses],
    })
    return df, costo

def run_desagregacion(prod_hh, factor_demanda=1.0):
    meses = MESES
    mdl = LpProblem("Desag",LpMinimize)
    X = {(p,t):LpVariable(f"X_{p}_{t}",lowBound=0) for p in PRODUCTOS for t in meses}
    I = {(p,t):LpVariable(f"I_{p}_{t}",lowBound=0) for p in PRODUCTOS for t in meses}
    S = {(p,t):LpVariable(f"S_{p}_{t}",lowBound=0) for p in PRODUCTOS for t in meses}
    mdl += lpSum(100000*I[p,t]+150000*S[p,t] for p in PRODUCTOS for t in meses),"FO"
    for idx,t in enumerate(meses):
        tp = meses[idx-1] if idx>0 else None
        mdl += (lpSum(HORAS_PRODUCTO[p]*X[p,t] for p in PRODUCTOS)<=prod_hh[t]), f"Cap_{t}"
        for p in PRODUCTOS:
            d = int(DEM_HISTORICA[p][idx]*factor_demanda)
            if idx==0: mdl += I[p,t]-S[p,t]==INV_INICIAL[p]+X[p,t]-d, f"Bal_{p}_{t}"
            else:       mdl += I[p,t]-S[p,t]==I[p,tp]-S[p,tp]+X[p,t]-d, f"Bal_{p}_{t}"
    mdl.solve(PULP_CBC_CMD(msg=False))
    out = {}
    for p in PRODUCTOS:
        rows=[]
        for idx,t in enumerate(meses):
            xv=round(X[p,t].varValue or 0,2)
            iv=round(I[p,t].varValue or 0,2)
            sv=round(S[p,t].varValue or 0,2)
            ini=INV_INICIAL[p] if idx==0 else round(I[p,meses[idx-1]].varValue or 0,2)
            rows.append({"Mes":t,"Demanda":int(DEM_HISTORICA[p][idx]*factor_demanda),
                         "Produccion":xv,"Inv_Ini":ini,"Inv_Fin":iv,"Backlog":sv})
        out[p]=pd.DataFrame(rows)
    return out

# ─── Simulación ──────────────────────────────────────────────────────────────

def run_simulacion(plan_unidades, cap_recursos=None, falla=False,
                   factor_t=1.0, tamano_lote=None, semilla=42):
    random.seed(semilla); np.random.seed(semilla)
    if cap_recursos is None: cap_recursos = CAPACIDAD_BASE.copy()
    if tamano_lote  is None: tamano_lote  = TAMANO_LOTE_BASE.copy()

    lotes_data, uso_rec, sensores = [], [], []

    def reg_uso(env, recursos, prod="", lid=""):
        ts=round(env.now,3)
        for nm,r in recursos.items():
            uso_rec.append({"tiempo":ts,"recurso":nm,"ocupados":r.count,
                             "cola":len(r.queue),"capacidad":r.capacity,"producto":prod})

    def sensor_horno(env, recursos):
        while True:
            ocp=recursos["horno"].count
            temp=round(np.random.normal(160+ocp*20,5),2)
            sensores.append({"tiempo":round(env.now,1),"temperatura":temp,
                              "horno_ocup":ocp,"horno_cola":len(recursos["horno"].queue)})
            yield env.timeout(10)

    def proceso_lote(env, lid, prod, tam, recursos):
        t0=env.now; esperas={}
        for etapa,rec_nm,tmin,tmax in RUTAS[prod]:
            escala=math.sqrt(tam/TAMANO_LOTE_BASE[prod])
            tp=random.uniform(tmin,tmax)*escala*factor_t
            if falla and rec_nm=="horno": tp+=random.uniform(10,30)
            reg_uso(env,recursos,prod,lid)
            t_ei=env.now
            with recursos[rec_nm].request() as req:
                yield req
                esperas[etapa]=round(env.now-t_ei,3)
                reg_uso(env,recursos,prod,lid)
                yield env.timeout(tp)
            reg_uso(env,recursos,prod,lid)
        lotes_data.append({
            "lote_id":lid,"producto":prod,"tamano":tam,
            "t_creacion":round(t0,3),"t_fin":round(env.now,3),
            "tiempo_sistema":round(env.now-t0,3),
            "total_espera":round(sum(esperas.values()),3),
        })

    env = simpy.Environment()
    recursos = {nm:simpy.Resource(env,capacity=cap) for nm,cap in cap_recursos.items()}
    env.process(sensor_horno(env,recursos))

    dur_mes = 44*4*60
    lotes=[]; ctr=[0]
    for prod,unid in plan_unidades.items():
        if unid<=0: continue
        tam=tamano_lote[prod]; n=math.ceil(unid/tam)
        tasa=dur_mes/max(n,1); ta=random.expovariate(1/max(tasa,1))
        rem=unid
        for _ in range(n):
            lotes.append((round(ta,2),prod,min(tam,int(rem))))
            rem-=tam; ta+=random.expovariate(1/max(tasa,1))
    lotes.sort(key=lambda x:x[0])

    def lanzador():
        for ta,prod,tam in lotes:
            yield env.timeout(max(ta-env.now,0))
            lid=f"{prod[:3].upper()}_{ctr[0]:04d}"; ctr[0]+=1
            env.process(proceso_lote(env,lid,prod,tam,recursos))
    env.process(lanzador())
    env.run(until=dur_mes*1.8)

    df_l = pd.DataFrame(lotes_data) if lotes_data else pd.DataFrame()
    df_u = pd.DataFrame(uso_rec)    if uso_rec    else pd.DataFrame()
    df_s = pd.DataFrame(sensores)   if sensores   else pd.DataFrame()
    return df_l, df_u, df_s

def calc_utilizacion(df_u):
    if df_u.empty: return pd.DataFrame()
    filas=[]
    for rec,grp in df_u.groupby("recurso"):
        grp=grp.sort_values("tiempo").reset_index(drop=True)
        cap=grp["capacidad"].iloc[0]; t=grp["tiempo"].values; ocp=grp["ocupados"].values
        if len(t)>1 and (t[-1]-t[0])>0:
            fn_trapz = np.trapezoid if hasattr(np,"trapezoid") else np.trapz
            area=fn_trapz(ocp,t); util=round(area/(cap*(t[-1]-t[0]))*100,2)
        else: util=0.0
        filas.append({"Recurso":rec,"Utilización_%":util,
                       "Cola Prom":round(grp["cola"].mean(),3),
                       "Cola Máx":int(grp["cola"].max()),
                       "Capacidad":int(cap),
                       "Cuello Botella":util>=80 or grp["cola"].mean()>0.5})
    return pd.DataFrame(filas).sort_values("Utilización_%",ascending=False).reset_index(drop=True)

def calc_kpis(df_l, plan):
    if df_l.empty: return pd.DataFrame()
    dur=(df_l["t_fin"].max()-df_l["t_creacion"].min())/60
    rows=[]
    for p in PRODUCTOS:
        sub=df_l[df_l["producto"]==p]
        if sub.empty: continue
        und=sub["tamano"].sum(); plan_und=plan.get(p,0)
        tp=round(und/max(dur,0.01),3)
        ct=round((sub["tiempo_sistema"]/sub["tamano"]).mean(),3)
        lt=round(sub["tiempo_sistema"].mean(),3)
        dem_avg=sum(DEM_HISTORICA[p])/12
        takt=round((44*4*60)/max(dem_avg/TAMANO_LOTE_BASE[p],1),2)
        wip=round(tp*(lt/60),2)
        rows.append({"Producto":p,"Und Producidas":und,"Plan":plan_und,
                      "Throughput (und/h)":tp,"Cycle Time (min/und)":ct,
                      "Lead Time (min/lote)":lt,"WIP Prom":wip,
                      "Takt Time (min/lote)":takt,
                      "Cumplimiento %":round(min(und/max(plan_und,1)*100,100),2)})
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# FIGURAS PLOTLY (tema oscuro personalizado)
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font         =dict(family="IBM Plex Mono, monospace", color="#e0d5c5", size=11),
    xaxis        =dict(gridcolor="#2a2a2a", zerolinecolor="#333"),
    yaxis        =dict(gridcolor="#2a2a2a", zerolinecolor="#333"),
    legend       =dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    margin       =dict(l=50,r=20,t=55,b=50),
    colorway     =["#E8A838","#4FC3F7","#81C784","#CE93D8","#FF8A65","#F06292"],
)

def apply_theme(fig, title="", height=400):
    fig.update_layout(**THEME, title=dict(text=title,x=0.5,
                      font=dict(size=15,color="#E8A838",family="Barlow Condensed, sans-serif")),
                      height=height)
    fig.update_xaxes(gridcolor="#222", zerolinecolor="#333")
    fig.update_yaxes(gridcolor="#222", zerolinecolor="#333")
    return fig

def fig_demanda():
    fig=go.Figure()
    for p in PRODUCTOS:
        fig.add_trace(go.Bar(x=MESES,y=DEM_HISTORICA[p],name=p.replace("_"," "),
                              marker_color=PROD_COLORS[p],opacity=0.85,
                              hovertemplate=f"<b>{p}</b><br>%{{x}}<br>%{{y:.0f}} und<extra></extra>"))
    fig.update_layout(**THEME,barmode="group",
                      title=dict(text="Demanda Histórica por Producto",x=0.5,
                                 font=dict(size=15,color="#E8A838",family="Barlow Condensed, sans-serif")),
                      xaxis_title="Mes",yaxis_title="Unidades",height=420,
                      legend=dict(bgcolor="rgba(0,0,0,0)",orientation="h",y=-0.28,x=0.5,xanchor="center"))
    return fig

def fig_heatmap_demanda():
    z=[[DEM_HISTORICA[p][i] for i in range(12)] for p in PRODUCTOS]
    fig=go.Figure(go.Heatmap(z=z,x=MESES,y=[p.replace("_"," ") for p in PRODUCTOS],
                              colorscale="YlOrBr",
                              hovertemplate="%{y}<br>%{x}<br>%{z:.0f} und<extra></extra>"))
    apply_theme(fig,"Mapa de Calor — Estacionalidad de la Demanda",360)
    return fig

def fig_agregacion(df_agr, costo):
    fig=go.Figure()
    fig.add_trace(go.Bar(x=df_agr["Mes"],y=df_agr["Inventario_Inicial_HH"],
                          name="Inv. Inicial (H-H)",marker_color="#5C6BC0",opacity=0.8))
    fig.add_trace(go.Bar(x=df_agr["Mes"],y=df_agr["Produccion_HH"],
                          name="Producción (H-H)",marker_color="#E8A838",opacity=0.85))
    fig.add_trace(go.Scatter(x=df_agr["Mes"],y=df_agr["Demanda_HH"],mode="lines+markers",
                              name="Demanda (H-H)",line=dict(color="#81C784",dash="dash",width=2.5),
                              marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=df_agr["Mes"],y=df_agr["Horas_Regulares"],mode="lines",
                              name="Cap. Regular",line=dict(color="#FF8A65",dash="dot",width=2)))
    fig.update_layout(**THEME,barmode="stack",height=420,
                      title=dict(text=f"Plan Agregado — Costo Óptimo: COP ${costo:,.0f}",
                                 x=0.5,font=dict(size=15,color="#E8A838",family="Barlow Condensed, sans-serif")),
                      xaxis_title="Mes",yaxis_title="Horas-Hombre (H-H)",
                      legend=dict(bgcolor="rgba(0,0,0,0)",orientation="h",y=-0.28,x=0.5,xanchor="center"))
    return fig

def fig_desagregacion(desag_dict, mes_sel):
    fig=make_subplots(rows=3,cols=2,subplot_titles=[p.replace("_"," ") for p in PRODUCTOS],
                      vertical_spacing=0.1,horizontal_spacing=0.08)
    for idx,p in enumerate(PRODUCTOS):
        r,c=idx//2+1,idx%2+1
        df=desag_dict[p]
        fig.add_trace(go.Bar(x=df["Mes"],y=df["Produccion"],name=p.replace("_"," "),
                              marker_color=PROD_COLORS[p],opacity=0.85,showlegend=False,
                              hovertemplate="%{x}<br>Prod: %{y:.0f} und<extra></extra>"),row=r,col=c)
        fig.add_trace(go.Scatter(x=df["Mes"],y=df["Demanda"],mode="lines+markers",
                                  name="Dem",line=dict(color="#81C784",dash="dash",width=1.5),
                                  marker=dict(size=5),showlegend=False),row=r,col=c)
        # Marca mes seleccionado
        mes_row=df[df["Mes"]==mes_sel]
        if not mes_row.empty:
            fig.add_trace(go.Scatter(x=[mes_sel],y=[mes_row["Produccion"].values[0]],
                                      mode="markers",marker=dict(size=12,color="#E8A838",
                                      symbol="star"),showlegend=False),row=r,col=c)
    fig.update_layout(**THEME,height=700,barmode="group",
                      title=dict(text="Desagregación por Producto (unidades/mes)",
                                 x=0.5,font=dict(size=15,color="#E8A838",family="Barlow Condensed, sans-serif")))
    for i in range(1,4):
        for j in range(1,3):
            fig.update_xaxes(gridcolor="#222",row=i,col=j)
            fig.update_yaxes(gridcolor="#222",title_text="und",row=i,col=j)
    return fig

def fig_utilizacion(df_u):
    df_ut=calc_utilizacion(df_u)
    if df_ut.empty: return go.Figure()
    colores=["#c0392b" if u>=80 else "#E8A838" if u>=60 else "#4FC3F7"
             for u in df_ut["Utilización_%"]]
    fig=make_subplots(rows=1,cols=2,
                      subplot_titles=["Utilización por Recurso (%)","Cola Promedio"])
    fig.add_trace(go.Bar(x=df_ut["Recurso"],y=df_ut["Utilización_%"],
                          marker_color=colores,
                          text=df_ut["Utilización_%"].apply(lambda v:f"{v:.1f}%"),
                          textposition="outside",showlegend=False,
                          hovertemplate="%{x}<br>Util: %{y:.2f}%<extra></extra>"),row=1,col=1)
    fig.add_trace(go.Bar(x=df_ut["Recurso"],y=df_ut["Cola Prom"],
                          marker_color="#CE93D8",
                          text=df_ut["Cola Prom"].apply(lambda v:f"{v:.2f}"),
                          textposition="outside",showlegend=False,
                          hovertemplate="%{x}<br>Cola: %{y:.2f}<extra></extra>"),row=1,col=2)
    fig.add_hline(y=80,line_dash="dash",line_color="#c0392b",
                  annotation_text="⚠ 80%",row=1,col=1)
    fig.update_layout(**THEME,height=400,
                      title=dict(text="Utilización de Recursos — Detección Cuellos de Botella",
                                 x=0.5,font=dict(size=15,color="#E8A838",family="Barlow Condensed, sans-serif")))
    fig.update_xaxes(gridcolor="#222"); fig.update_yaxes(gridcolor="#222")
    return fig

def fig_kpis_radar(df_kpi):
    if df_kpi.empty: return go.Figure()
    cats=["Throughput (und/h)","Cycle Time (min/und)","Lead Time (min/lote)","WIP Prom","Cumplimiento %"]
    fig=go.Figure()
    for _,row in df_kpi.iterrows():
        vals=[row.get(c,0) for c in cats]
        # Normalizar a 0-100 para radar
        maxv=[max(df_kpi[c].max(),0.01) for c in cats]
        norm=[round(v/m*100,1) for v,m in zip(vals,maxv)]
        norm.append(norm[0])
        fig.add_trace(go.Scatterpolar(
            r=norm, theta=cats+[cats[0]],
            name=row["Producto"].replace("_"," "),
            fill="toself", opacity=0.6,
            line=dict(color=PROD_COLORS.get(row["Producto"],"#aaa"),width=2),
        ))
    fig.update_layout(**THEME,height=420,
                      polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(visible=True,gridcolor="#333",tickcolor="#555"),
                                 angularaxis=dict(gridcolor="#333")),
                      title=dict(text="Radar de KPIs por Producto (normalizado)",
                                 x=0.5,font=dict(size=15,color="#E8A838",family="Barlow Condensed, sans-serif")),
                      legend=dict(bgcolor="rgba(0,0,0,0)",orientation="h",y=-0.15,x=0.5,xanchor="center"))
    return fig

def fig_sensores(df_s):
    if df_s.empty: return go.Figure()
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,
                      subplot_titles=["Temperatura del Horno (°C)","Ocupación del Horno"])
    fig.add_trace(go.Scatter(x=df_s["tiempo"],y=df_s["temperatura"],
                              mode="lines",name="Temp",
                              line=dict(color="#FF8A65",width=1.5),
                              hovertemplate="t=%{x:.0f} min<br>%{y:.1f}°C<extra></extra>"),row=1,col=1)
    fig.add_hline(y=200,line_dash="dash",line_color="#c0392b",
                  annotation_text="Límite 200°C",row=1,col=1)
    fig.add_trace(go.Scatter(x=df_s["tiempo"],y=df_s["horno_ocup"],
                              mode="lines",name="Ocupación",
                              fill="tozeroy",fillcolor="rgba(79,195,247,0.12)",
                              line=dict(color="#4FC3F7",width=1.5)),row=2,col=1)
    fig.update_layout(**THEME,height=460,
                      title=dict(text="Sensores Virtuales — Monitor del Horno en Tiempo Real",
                                 x=0.5,font=dict(size=15,color="#E8A838",family="Barlow Condensed, sans-serif")))
    fig.update_xaxes(gridcolor="#222",title_text="Tiempo simulado (min)",row=2,col=1)
    fig.update_yaxes(gridcolor="#222",title_text="°C",row=1,col=1)
    fig.update_yaxes(gridcolor="#222",title_text="Estaciones",row=2,col=1)
    return fig

def fig_colas(df_u):
    if df_u.empty: return go.Figure()
    fig=go.Figure()
    paleta=["#E8A838","#4FC3F7","#81C784","#CE93D8","#FF8A65","#F06292"]
    for i,(rec,grp) in enumerate(df_u.groupby("recurso")):
        grp=grp.sort_values("tiempo")
        fig.add_trace(go.Scatter(x=grp["tiempo"],y=grp["cola"],mode="lines",
                                  name=rec,line=dict(color=paleta[i%len(paleta)],width=1.5),
                                  hovertemplate=f"<b>{rec}</b><br>t=%{{x:.0f}} min<br>Cola=%{{y}}<extra></extra>"))
    apply_theme(fig,"Evolución de Colas por Recurso (SimPy)",400)
    fig.update_xaxes(title_text="Tiempo simulado (min)")
    fig.update_yaxes(title_text="Tamaño de cola")
    fig.update_layout(legend=dict(bgcolor="rgba(0,0,0,0)",orientation="h",
                                   y=-0.22,x=0.5,xanchor="center"))
    return fig

def fig_gantt(df_l, n=80):
    if df_l.empty: return go.Figure()
    sub=df_l.head(n).copy().reset_index(drop=True)
    fig=go.Figure()
    for _,row in sub.iterrows():
        col=PROD_COLORS.get(row["producto"],"#aaa")
        fig.add_trace(go.Bar(
            x=[row["tiempo_sistema"]], y=[row["lote_id"]],
            base=[row["t_creacion"]], orientation="h",
            marker_color=col, opacity=0.8,
            hovertemplate=(f"<b>{row['producto']}</b><br>"
                           f"Lote: {row['lote_id']}<br>"
                           f"Inicio: {row['t_creacion']:.0f} min<br>"
                           f"Duración: {row['tiempo_sistema']:.1f} min<extra></extra>"),
            showlegend=False,
        ))
    # Leyenda manual
    for p,c in PROD_COLORS.items():
        fig.add_trace(go.Bar(x=[None],y=[None],marker_color=c,name=p.replace("_"," "),showlegend=True))
    apply_theme(fig,"Diagrama de Gantt — Lotes de Producción",max(350,len(sub)*7))
    fig.update_layout(barmode="overlay",
                      xaxis_title="Tiempo simulado (min)",yaxis_title="Lote ID",
                      legend=dict(bgcolor="rgba(0,0,0,0)",orientation="h",
                                   y=-0.18,x=0.5,xanchor="center"))
    return fig

def fig_comparacion(resultados_esc):
    if not resultados_esc: return go.Figure()
    filas=[]
    for nm,res in resultados_esc.items():
        dk=res.get("kpis",pd.DataFrame())
        du=res.get("util",pd.DataFrame())
        if dk.empty: continue
        fila={"Escenario":nm}
        for col in ["Throughput (und/h)","Lead Time (min/lote)","WIP Prom","Cumplimiento %"]:
            if col in dk.columns: fila[col]=round(dk[col].mean(),2)
        if not du.empty and "Utilización_%" in du.columns:
            fila["Util Máx %"]=round(du["Utilización_%"].max(),2)
        filas.append(fila)
    df=pd.DataFrame(filas)

    metricas=[("Throughput (und/h)","Throughput (und/h)"),
              ("Lead Time (min/lote)","Lead Time (min)"),
              ("Cumplimiento %","Cumplimiento (%)"),
              ("Util Máx %","Util. Máx (%)")]
    fig=make_subplots(rows=2,cols=2,subplot_titles=[m[1] for m in metricas])
    paleta=["#E8A838","#4FC3F7","#81C784","#CE93D8","#FF8A65","#F06292","#80DEEA"]
    for i,(col,titulo) in enumerate(metricas):
        r,c=i//2+1,i%2+1
        if col not in df.columns: continue
        fig.add_trace(go.Bar(
            x=df["Escenario"],y=df[col],
            marker_color=[paleta[j%len(paleta)] for j in range(len(df))],
            text=df[col].apply(lambda v:f"{v:.2f}"),textposition="outside",
            showlegend=False,
            hovertemplate="%{x}<br>"+col+": %{y:.2f}<extra></extra>",
        ),row=r,col=c)
    fig.update_layout(**THEME,height=520,
                      title=dict(text="Comparación de Escenarios What-If",
                                 x=0.5,font=dict(size=15,color="#E8A838",family="Barlow Condensed, sans-serif")))
    fig.update_xaxes(gridcolor="#222",tickangle=25)
    fig.update_yaxes(gridcolor="#222")
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# COMPONENTES UI
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

# CSS global inyectado
EXTERNAL_CSS = [
    "https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700&family=IBM+Plex+Mono:wght@300;400;500&display=swap",
    dbc.themes.CYBORG,
]

CARD_STYLE = {
    "background":"#111418",
    "border":"1px solid #2a2a2a",
    "borderRadius":"6px",
    "padding":"16px",
}
LABEL_STYLE = {
    "color":"#8a8a8a",
    "fontSize":"10px",
    "fontFamily":"IBM Plex Mono, monospace",
    "letterSpacing":"0.12em",
    "textTransform":"uppercase",
    "marginBottom":"4px",
}
INPUT_STYLE = {
    "background":"#0d1117",
    "color":"#e0d5c5",
    "border":"1px solid #333",
    "borderRadius":"4px",
    "fontFamily":"IBM Plex Mono, monospace",
    "fontSize":"12px",
}

def kpi_card(titulo, valor, unidad="", color="#E8A838", icon="◈"):
    return html.Div([
        html.Div(icon+" "+titulo, style={**LABEL_STYLE,"color":"#666"}),
        html.Div([
            html.Span(str(valor), style={"fontSize":"26px","fontWeight":"600",
                                          "color":color,"fontFamily":"Barlow Condensed, sans-serif"}),
            html.Span(" "+unidad, style={"fontSize":"11px","color":"#666","marginLeft":"4px",
                                          "fontFamily":"IBM Plex Mono, monospace"}),
        ]),
    ], style={**CARD_STYLE,"minWidth":"140px"})

def seccion_titulo(texto, sub=""):
    return html.Div([
        html.H4(texto, style={"fontFamily":"Barlow Condensed, sans-serif",
                               "fontWeight":"700","letterSpacing":"0.06em",
                               "color":"#E8A838","margin":"0 0 2px 0","fontSize":"20px"}),
        html.P(sub, style={"color":"#555","fontSize":"11px","margin":"0",
                            "fontFamily":"IBM Plex Mono, monospace"}) if sub else None,
    ], style={"borderLeft":"3px solid #E8A838","paddingLeft":"12px","marginBottom":"20px"})

def tabla_dash(df, id_tabla, page_size=12):
    if df.empty: return html.Div("Sin datos", style={"color":"#555"})
    return dash_table.DataTable(
        id=id_tabla,
        columns=[{"name":c,"id":c} for c in df.columns],
        data=df.round(3).to_dict("records"),
        page_size=page_size,
        style_table={"overflowX":"auto"},
        style_header={"backgroundColor":"#0d1117","color":"#E8A838",
                       "fontFamily":"IBM Plex Mono, monospace","fontSize":"10px",
                       "border":"1px solid #222","letterSpacing":"0.08em"},
        style_cell={"backgroundColor":"#111418","color":"#c0b8a8",
                     "fontFamily":"IBM Plex Mono, monospace","fontSize":"11px",
                     "border":"1px solid #1e1e1e","padding":"6px 10px","textAlign":"right"},
        style_data_conditional=[
            {"if":{"row_index":"odd"},"backgroundColor":"#0e1216"},
        ],
    )

# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# APP DASH
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=EXTERNAL_CSS,
    suppress_callback_exceptions=True,
    title="Gemelo Digital — Dora del Hoyo",
)

# ─── Barra de navegación lateral ─────────────────────────────────────────────

NAV_ITEMS = [
    ("01","DEMANDA",      "tab-demanda"),
    ("02","PLANEACIÓN",   "tab-agregacion"),
    ("03","DESAGREGACIÓN","tab-desag"),
    ("04","SIMULACIÓN",   "tab-sim"),
    ("05","KPIs",         "tab-kpis"),
    ("06","SENSORES",     "tab-sensores"),
    ("07","ESCENARIOS",   "tab-escenarios"),
]

def nav_link(num, label, tab_id):
    return html.Button(
        [html.Span(num, style={"fontSize":"9px","color":"#E8A838","marginRight":"8px",
                                "fontFamily":"IBM Plex Mono, monospace"}),
         html.Span(label, style={"fontSize":"12px","letterSpacing":"0.1em"})],
        id=f"btn-{tab_id}",
        n_clicks=0,
        style={"background":"transparent","border":"none","color":"#8a8a8a",
               "width":"100%","textAlign":"left","padding":"10px 16px",
               "cursor":"pointer","fontFamily":"Barlow Condensed, sans-serif",
               "fontWeight":"600","transition":"color 0.2s"},
        className="nav-btn",
    )

sidebar = html.Div([
    # Logo / título
    html.Div([
        html.Div("◈", style={"fontSize":"28px","color":"#E8A838","lineHeight":"1"}),
        html.Div("DORA DEL HOYO", style={"fontSize":"11px","fontWeight":"700",
                                          "letterSpacing":"0.18em","color":"#e0d5c5",
                                          "fontFamily":"Barlow Condensed, sans-serif"}),
        html.Div("GEMELO DIGITAL", style={"fontSize":"9px","color":"#E8A838",
                                           "letterSpacing":"0.22em",
                                           "fontFamily":"IBM Plex Mono, monospace"}),
    ], style={"padding":"24px 16px 20px","borderBottom":"1px solid #1e1e1e","marginBottom":"12px"}),

    # Botones de navegación
    html.Div([nav_link(n,l,t) for n,l,t in NAV_ITEMS]),

    # Indicador de estado
    html.Div(id="sidebar-status", style={"padding":"16px","marginTop":"auto",
                                          "borderTop":"1px solid #1e1e1e","position":"absolute",
                                          "bottom":"0","width":"100%"}),
], style={"width":"200px","minHeight":"100vh","background":"#0a0d11",
          "borderRight":"1px solid #1a1a1a","display":"flex","flexDirection":"column",
          "position":"fixed","top":"0","left":"0","zIndex":"100"})

# ─── Panel de configuración ───────────────────────────────────────────────────

panel_config = html.Div([
    html.Div([
        seccion_titulo("CONFIGURACIÓN GLOBAL", "parámetros del modelo"),
        dbc.Row([
            dbc.Col([
                html.Div("MES A SIMULAR", style=LABEL_STYLE),
                dcc.Dropdown(
                    id="dd-mes", options=[{"label":m,"value":i} for i,m in enumerate(MESES)],
                    value=1, clearable=False,
                    style={**INPUT_STYLE,"minWidth":"160px"},
                    className="custom-dropdown",
                ),
            ], width=3),
            dbc.Col([
                html.Div("FACTOR DEMANDA", style=LABEL_STYLE),
                dcc.Slider(id="sl-demanda",min=0.5,max=2.0,step=0.1,value=1.0,
                           marks={0.5:"0.5×",1.0:"1×",1.5:"1.5×",2.0:"2×"},
                           tooltip={"placement":"top","always_visible":True}),
            ], width=3),
            dbc.Col([
                html.Div("CAPACIDAD HORNO (estaciones)", style=LABEL_STYLE),
                dcc.Slider(id="sl-horno",min=1,max=6,step=1,value=3,
                           marks={i:str(i) for i in range(1,7)},
                           tooltip={"placement":"top","always_visible":True}),
            ], width=3),
            dbc.Col([
                html.Div("OPCIONES", style=LABEL_STYLE),
                dbc.Checklist(
                    id="chk-opciones",
                    options=[{"label":"Falla en Horno","value":"falla"},
                              {"label":"Doble Turno (−20% tiempos)","value":"turno"}],
                    value=[], switch=True,
                    style={"color":"#c0b8a8","fontSize":"12px",
                           "fontFamily":"IBM Plex Mono, monospace"},
                ),
            ], width=3),
        ]),
        html.Div(style={"height":"12px"}),
        dbc.Row([
            dbc.Col([
                html.Button("▶  EJECUTAR PIPELINE",id="btn-run",n_clicks=0,
                    style={"background":"#E8A838","color":"#0a0d11","border":"none",
                           "padding":"10px 28px","fontFamily":"Barlow Condensed, sans-serif",
                           "fontWeight":"700","fontSize":"14px","letterSpacing":"0.12em",
                           "borderRadius":"4px","cursor":"pointer","width":"100%"}),
            ], width=3),
            dbc.Col([
                html.Div(id="run-status",style={"color":"#81C784","fontSize":"12px",
                                                 "fontFamily":"IBM Plex Mono, monospace",
                                                 "padding":"10px 0"}),
            ], width=9),
        ]),
    ], style={**CARD_STYLE,"marginBottom":"0"}),
], style={"padding":"16px 24px 0"})

# ─── Tabs de contenido ────────────────────────────────────────────────────────

content_area = html.Div([
    dcc.Loading(id="loading-main", type="dot", color="#E8A838",
                children=html.Div(id="tab-content")),
], style={"padding":"16px 24px"})

# ─── Store (memoria compartida) ───────────────────────────────────────────────

stores = html.Div([
    dcc.Store(id="store-active-tab",  data="tab-demanda"),
    dcc.Store(id="store-agr",        data=None),
    dcc.Store(id="store-desag",      data=None),
    dcc.Store(id="store-sim",        data=None),
    dcc.Store(id="store-util",       data=None),
    dcc.Store(id="store-kpis",       data=None),
    dcc.Store(id="store-sensores",   data=None),
    dcc.Store(id="store-plan-mes",   data=None),
    dcc.Store(id="store-esc",        data={}),
])

# ─── Layout principal ─────────────────────────────────────────────────────────

app.layout = html.Div([
    stores,
    sidebar,
    html.Div([
        # Header
        html.Div([
            html.Div([
                html.Span("●", style={"color":"#E8A838","marginRight":"8px"}),
                html.Span("PLANTA DE PRODUCCIÓN — GEMELO DIGITAL",
                          style={"fontFamily":"IBM Plex Mono, monospace","fontSize":"11px",
                                 "letterSpacing":"0.18em","color":"#555"}),
            ]),
            html.Div(id="header-tab-name",
                     style={"fontFamily":"Barlow Condensed, sans-serif","fontWeight":"700",
                             "fontSize":"28px","color":"#e0d5c5","letterSpacing":"0.04em"}),
        ], style={"padding":"20px 24px 0","borderBottom":"1px solid #1a1a1a",
                   "marginBottom":"0","background":"#0a0d11"}),

        panel_config,

        html.Hr(style={"borderColor":"#1a1a1a","margin":"16px 0"}),

        content_area,
    ], style={"marginLeft":"200px","minHeight":"100vh","background":"#0d1117"}),


], style={"fontFamily":"IBM Plex Mono, monospace","color":"#e0d5c5"})


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

# ── Navegación lateral → active tab ──────────────────────────────────────────

@app.callback(
    Output("store-active-tab","data"),
    Output("header-tab-name","children"),
    [Input(f"btn-{t}","n_clicks") for _,l,t in NAV_ITEMS],
    prevent_initial_call=True,
)
def cambiar_tab(*args):
    ctx = callback_context
    if not ctx.triggered: return dash.no_update, dash.no_update
    btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
    tab_id = btn_id.replace("btn-","")
    label  = next(l for _,l,t in NAV_ITEMS if t==tab_id)
    return tab_id, label

# ── Ejecutar pipeline ─────────────────────────────────────────────────────────

@app.callback(
    Output("store-agr","data"),
    Output("store-desag","data"),
    Output("store-sim","data"),
    Output("store-util","data"),
    Output("store-kpis","data"),
    Output("store-sensores","data"),
    Output("store-plan-mes","data"),
    Output("run-status","children"),
    Input("btn-run","n_clicks"),
    State("dd-mes","value"),
    State("sl-demanda","value"),
    State("sl-horno","value"),
    State("chk-opciones","value"),
    prevent_initial_call=True,
)
def ejecutar_pipeline(n_clicks, mes_idx, factor_dem, cap_horno, opciones):
    if not n_clicks:
        return (None,)*7 + ("",)
    try:
        # 1. Demanda + Agregación
        dem_h = _dem_horas({m: factor_dem for m in MESES})
        df_agr, costo = run_agregacion(dem_h)
        prod_hh = dict(zip(df_agr["Mes"], df_agr["Produccion_HH"]))

        # 2. Desagregación
        desag = run_desagregacion(prod_hh, factor_dem)

        # 3. Plan del mes
        mes_nm = MESES[mes_idx]
        plan_mes = {
            p: int(desag[p].loc[desag[p]["Mes"]==mes_nm,"Produccion"].values[0])
            for p in PRODUCTOS
        }

        # 4. Simulación
        cap_rec = {**CAPACIDAD_BASE, "horno": int(cap_horno)}
        factor_t = 0.80 if "turno" in (opciones or []) else 1.0
        falla    = "falla" in (opciones or [])
        df_l, df_u, df_s = run_simulacion(plan_mes, cap_rec, falla, factor_t)

        # 5. KPIs / Utilización
        df_kpi = calc_kpis(df_l, plan_mes)
        df_ut  = calc_utilizacion(df_u)

        # Serializar
        agr_json   = df_agr.to_json()
        desag_json = {p: df.to_json() for p,df in desag.items()}
        sim_json   = df_l.to_json()   if not df_l.empty else "{}"
        util_json  = df_u.to_json()   if not df_u.empty else "{}"
        kpi_json   = df_kpi.to_json() if not df_kpi.empty else "{}"
        sen_json   = df_s.to_json()   if not df_s.empty else "{}"

        status = (f"✓ Pipeline completado — {len(df_l)} lotes simulados | "
                  f"Mes: {mes_nm} | Costo agregado: COP ${costo:,.0f}")
        return agr_json, desag_json, sim_json, util_json, kpi_json, sen_json, plan_mes, status

    except Exception as e:
        return (None,)*7 + (f"✗ Error: {str(e)}",)

# ── Correr escenario individual ───────────────────────────────────────────────

@app.callback(
    Output("store-esc","data"),
    Input("btn-esc","n_clicks"),
    State("dd-esc","value"),
    State("store-plan-mes","data"),
    State("dd-mes","value"),
    State("store-esc","data"),
    prevent_initial_call=True,
)
def correr_escenario_cb(n, escenarios_sel, plan_mes, mes_idx, esc_store):
    if not n or not plan_mes or not escenarios_sel:
        return esc_store or {}

    # Definición de escenarios
    ESC_DEF = {
        "base":{"fd":1.0,"falla":False,"ft":1.0,"dh":0},
        "demanda_20":{"fd":1.2,"falla":False,"ft":1.0,"dh":0},
        "falla_horno":{"fd":1.0,"falla":True,"ft":1.0,"dh":0},
        "red_cap":{"fd":1.0,"falla":False,"ft":1.0,"dh":-1},
        "doble_turno":{"fd":1.0,"falla":False,"ft":0.80,"dh":0},
        "lote_grande":{"fd":1.0,"falla":False,"ft":1.0,"dh":0,"fl":1.5},
        "optimizado":{"fd":1.0,"falla":False,"ft":0.85,"dh":1},
    }

    resultado = dict(esc_store or {})
    for nm in escenarios_sel:
        cfg = ESC_DEF.get(nm, ESC_DEF["base"])
        plan_aj = {p: max(int(u*cfg["fd"]),0) for p,u in plan_mes.items()}
        cap_r   = {**CAPACIDAD_BASE, "horno": max(CAPACIDAD_BASE["horno"]+cfg.get("dh",0),1)}
        tam_l   = {p: max(int(t*cfg.get("fl",1.0)),1) for p,t in TAMANO_LOTE_BASE.items()}
        df_l,df_u,_ = run_simulacion(plan_aj,cap_r,cfg["falla"],cfg["ft"],tam_l)
        dk = calc_kpis(df_l,plan_aj)
        du = calc_utilizacion(df_u)
        resultado[nm] = {
            "kpis": dk.to_json() if not dk.empty else "{}",
            "util": du.to_json() if not du.empty else "{}",
        }
    return resultado

# ── Renderizar contenido del tab activo ───────────────────────────────────────

@app.callback(
    Output("tab-content","children"),
    Input("store-active-tab","data"),
    State("store-agr","data"),
    State("store-desag","data"),
    State("store-sim","data"),
    State("store-util","data"),
    State("store-kpis","data"),
    State("store-sensores","data"),
    State("store-plan-mes","data"),
    State("store-esc","data"),
    State("dd-mes","value"),
)
def render_tab(tab, agr_j, desag_j, sim_j, util_j, kpi_j, sen_j, plan_mes, esc_store, mes_idx):

    # ── DEMANDA ───────────────────────────────────────────────────────────────
    if tab == "tab-demanda":
        total_anual = sum(sum(DEM_HISTORICA[p]) for p in PRODUCTOS)
        pico_mes    = max(MESES, key=lambda m: sum(DEM_HISTORICA[p][MESES.index(m)] for p in PRODUCTOS))
        return html.Div([
            seccion_titulo("DEMANDA HISTÓRICA", "análisis de estacionalidad por producto"),
            dbc.Row([
                dbc.Col(kpi_card("Total Anual",f"{total_anual:,}","unidades","#E8A838","◈"),width="auto"),
                dbc.Col(kpi_card("Productos",len(PRODUCTOS),"","#4FC3F7","◈"),width="auto"),
                dbc.Col(kpi_card("Mes Pico",pico_mes,"","#81C784","◈"),width="auto"),
                dbc.Col(kpi_card("Horizonte","12","meses","#CE93D8","◈"),width="auto"),
            ], className="g-3 mb-4"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_demanda(), config={"displayModeBar":False}), width=8),
                dbc.Col(dcc.Graph(figure=fig_heatmap_demanda(), config={"displayModeBar":False}), width=4),
            ], className="g-3 mb-4"),
            seccion_titulo("TABLA DE DEMANDA HISTÓRICA","unidades por mes"),
            tabla_dash(pd.DataFrame(DEM_HISTORICA, index=MESES).reset_index().rename(columns={"index":"Mes"}),
                       "tbl-demanda"),
        ])

    # ── PLANEACIÓN AGREGADA ───────────────────────────────────────────────────
    if tab == "tab-agregacion":
        if not agr_j:
            return _no_data_msg()
        df_agr = pd.read_json(agr_j)
        costo  = df_agr["Produccion_HH"].sum() * 4310
        return html.Div([
            seccion_titulo("PLANEACIÓN AGREGADA","optimización en horas-hombre (PuLP / CBC)"),
            dbc.Row([
                dbc.Col(kpi_card("Prod. Total HH",f"{df_agr['Produccion_HH'].sum():,.0f}","H-H","#E8A838"),width="auto"),
                dbc.Col(kpi_card("Backlog Total",f"{df_agr['Backlog_HH'].sum():,.1f}","H-H","#FF8A65"),width="auto"),
                dbc.Col(kpi_card("H. Extra Total",f"{df_agr['Horas_Extras'].sum():,.1f}","H-H","#4FC3F7"),width="auto"),
                dbc.Col(kpi_card("Contrataciones",f"{df_agr['Contratacion'].sum():,.0f}","","#81C784"),width="auto"),
            ], className="g-3 mb-4"),
            dcc.Graph(figure=fig_agregacion(df_agr, costo), config={"displayModeBar":False}),
            html.Div(style={"height":"16px"}),
            seccion_titulo("TABLA DETALLADA","plan mensual en H-H"),
            tabla_dash(df_agr, "tbl-agr"),
        ])

    # ── DESAGREGACIÓN ─────────────────────────────────────────────────────────
    if tab == "tab-desag":
        if not desag_j:
            return _no_data_msg()
        desag_dict = {p: pd.read_json(v) for p,v in desag_j.items()}
        mes_nm     = MESES[mes_idx or 1]
        total_und  = sum(desag_dict[p]["Produccion"].sum() for p in PRODUCTOS)
        return html.Div([
            seccion_titulo("DESAGREGACIÓN DEL PLAN","unidades por producto y mes"),
            dbc.Row([
                dbc.Col(kpi_card("Total Unidades",f"{total_und:,.0f}","und","#E8A838"),width="auto"),
                *[dbc.Col(kpi_card(
                    p.replace("_"," ").replace("Mantecadas ","Mant. "),
                    f"{desag_dict[p]['Produccion'].sum():,.0f}","und",
                    PROD_COLORS[p],
                ), width="auto") for p in PRODUCTOS],
            ], className="g-3 mb-4"),
            dcc.Graph(figure=fig_desagregacion(desag_dict, mes_nm), config={"displayModeBar":False}),
        ])

    # ── SIMULACIÓN ────────────────────────────────────────────────────────────
    if tab == "tab-sim":
        if not sim_j or sim_j == "{}":
            return _no_data_msg()
        df_l = pd.read_json(sim_j)
        n_lotes  = len(df_l)
        dur_prom = round(df_l["tiempo_sistema"].mean(),1) if not df_l.empty else 0
        t_total  = round(df_l["t_fin"].max(),1)           if not df_l.empty else 0
        return html.Div([
            seccion_titulo("SIMULACIÓN DE EVENTOS DISCRETOS","SimPy — flujo de lotes por recurso"),
            dbc.Row([
                dbc.Col(kpi_card("Lotes Simulados",n_lotes,"","#E8A838"),width="auto"),
                dbc.Col(kpi_card("Tiempo Total",f"{t_total:,.0f}","min","#4FC3F7"),width="auto"),
                dbc.Col(kpi_card("Duración Prom Lote",f"{dur_prom:,.1f}","min","#81C784"),width="auto"),
                dbc.Col(kpi_card("Productos",len(df_l["producto"].unique()),"","#CE93D8"),width="auto"),
            ], className="g-3 mb-4"),
            dcc.Graph(figure=fig_gantt(df_l, n=80), config={"displayModeBar":False}),
            html.Div(style={"height":"16px"}),
            html.Div([
                dcc.Graph(figure=fig_colas(pd.read_json(util_j)) if util_j and util_j!="{}" else go.Figure(),
                          config={"displayModeBar":False})
            ]),
            html.Div(style={"height":"16px"}),
            seccion_titulo("REGISTRO DE LOTES","primeros 200 resultados"),
            tabla_dash(df_l.head(200)[["lote_id","producto","tamano","t_creacion","t_fin",
                                       "tiempo_sistema","total_espera"]], "tbl-lotes"),
        ])

    # ── KPIs ──────────────────────────────────────────────────────────────────
    if tab == "tab-kpis":
        if not kpi_j or kpi_j == "{}":
            return _no_data_msg()
        df_kpi = pd.read_json(kpi_j)
        df_ut  = pd.read_json(util_j) if util_j and util_j!="{}" else pd.DataFrame()
        return html.Div([
            seccion_titulo("KPIs & CUELLOS DE BOTELLA","métricas de desempeño operativo"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_kpis_radar(df_kpi), config={"displayModeBar":False}), width=6),
                dbc.Col(dcc.Graph(figure=fig_utilizacion(df_ut),  config={"displayModeBar":False}), width=6),
            ], className="g-3 mb-4"),
            seccion_titulo("TABLA KPIs POR PRODUCTO"),
            tabla_dash(df_kpi,"tbl-kpis",page_size=10),
            html.Div(style={"height":"20px"}),
            seccion_titulo("UTILIZACIÓN DE RECURSOS"),
            tabla_dash(calc_utilizacion(df_ut),"tbl-util",page_size=10) if not df_ut.empty else html.Div(),
        ])

    # ── SENSORES ──────────────────────────────────────────────────────────────
    if tab == "tab-sensores":
        if not sen_j or sen_j == "{}":
            return _no_data_msg()
        df_s = pd.read_json(sen_j)
        t_max  = round(df_s["temperatura"].max(),1) if not df_s.empty else 0
        t_min  = round(df_s["temperatura"].min(),1) if not df_s.empty else 0
        t_prom = round(df_s["temperatura"].mean(),1) if not df_s.empty else 0
        return html.Div([
            seccion_titulo("SENSORES VIRTUALES","monitoreo en tiempo real — horno"),
            dbc.Row([
                dbc.Col(kpi_card("Temp. Máxima",t_max,"°C","#FF8A65"),width="auto"),
                dbc.Col(kpi_card("Temp. Mínima",t_min,"°C","#4FC3F7"),width="auto"),
                dbc.Col(kpi_card("Temp. Promedio",t_prom,"°C","#E8A838"),width="auto"),
                dbc.Col(kpi_card("Lecturas",len(df_s),"","#81C784"),width="auto"),
            ], className="g-3 mb-4"),
            dcc.Graph(figure=fig_sensores(df_s), config={"displayModeBar":False}),
        ])

    # ── ESCENARIOS WHAT-IF ────────────────────────────────────────────────────
    if tab == "tab-escenarios":
        esc_opts = [
            {"label":"Base","value":"base"},
            {"label":"Demanda +20%","value":"demanda_20"},
            {"label":"Falla Horno","value":"falla_horno"},
            {"label":"Reducir Capacidad Horno","value":"red_cap"},
            {"label":"Doble Turno (−20% tiempos)","value":"doble_turno"},
            {"label":"Lotes +50%","value":"lote_grande"},
            {"label":"Optimizado (+1 horno, −15% tiempos)","value":"optimizado"},
        ]
        fig_comp = go.Figure()
        if esc_store:
            res_parsed = {
                nm: {"kpis": pd.read_json(v["kpis"]) if v.get("kpis","{}") != "{}" else pd.DataFrame(),
                     "util": pd.read_json(v["util"]) if v.get("util","{}") != "{}" else pd.DataFrame()}
                for nm,v in esc_store.items()
            }
            fig_comp = fig_comparacion(res_parsed)

        return html.Div([
            seccion_titulo("ANÁLISIS DE ESCENARIOS WHAT-IF","evaluación comparativa de estrategias"),
            html.Div([
                html.Div("SELECCIONAR ESCENARIOS", style=LABEL_STYLE),
                dcc.Checklist(
                    id="dd-esc",
                    options=esc_opts,
                    value=["base","demanda_20"],
                    inline=True,
                    style={"color":"#c0b8a8","fontSize":"12px",
                           "fontFamily":"IBM Plex Mono, monospace","gap":"16px"},
                    labelStyle={"marginRight":"20px","display":"inline-block"},
                ),
                html.Div(style={"height":"12px"}),
                html.Button("▶ CORRER ESCENARIOS SELECCIONADOS", id="btn-esc", n_clicks=0,
                    style={"background":"#1e2a3a","color":"#4FC3F7","border":"1px solid #4FC3F7",
                           "padding":"8px 20px","fontFamily":"Barlow Condensed, sans-serif",
                           "fontWeight":"700","fontSize":"13px","letterSpacing":"0.1em",
                           "borderRadius":"4px","cursor":"pointer"}),
            ], style={**CARD_STYLE,"marginBottom":"20px"}),
            dcc.Graph(id="fig-comp",figure=fig_comp, config={"displayModeBar":False}),
        ])

    # Default
    return html.Div("Selecciona una sección en el menú lateral.",
                    style={"color":"#555","padding":"40px","fontFamily":"IBM Plex Mono, monospace"})


# Callback para actualizar fig comparación cuando cambia el store de escenarios
@app.callback(
    Output("fig-comp","figure"),
    Input("store-esc","data"),
    prevent_initial_call=True,
)
def update_fig_comp(esc_store):
    if not esc_store: return go.Figure()
    res_parsed = {
        nm: {"kpis": pd.read_json(v["kpis"]) if v.get("kpis","{}") != "{}" else pd.DataFrame(),
             "util": pd.read_json(v["util"]) if v.get("util","{}") != "{}" else pd.DataFrame()}
        for nm,v in esc_store.items()
    }
    return fig_comparacion(res_parsed)


def _no_data_msg():
    return html.Div([
        html.Div("◈", style={"fontSize":"48px","color":"#2a2a2a","textAlign":"center","padding":"40px 0 8px"}),
        html.Div("Ejecuta el pipeline primero",
                 style={"textAlign":"center","color":"#444","fontSize":"13px",
                        "fontFamily":"IBM Plex Mono, monospace"}),
        html.Div("Configura los parámetros y presiona ▶ EJECUTAR PIPELINE",
                 style={"textAlign":"center","color":"#333","fontSize":"11px",
                        "fontFamily":"IBM Plex Mono, monospace","marginTop":"6px"}),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═"*60)
    print("  GEMELO DIGITAL — DORA DEL HOYO")
    print("  Dashboard Dash disponible en http://127.0.0.1:8050")
    print("═"*60 + "\n")
    app.run(debug=False, port=8050, host="0.0.0.0")
