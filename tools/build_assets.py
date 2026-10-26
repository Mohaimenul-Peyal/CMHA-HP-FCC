"""Rebuild CMHA manuscript figures and tables from the frozen evidence package."""
from __future__ import annotations

import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'data/paper_data.json').read_text())
FOLDS = json.loads((ROOT / 'data/selected_folds.json').read_text())
PRIMARY = DATA['primary']
PRIMARY_MAP = {row['benchmark']: row for row in PRIMARY}
ABLATIONS = DATA['ablations']
ORDER = list(dict.fromkeys(row['variant'] for row in ABLATIONS))
LABELS = dict(zip(ORDER, ('Full CMHA', 'Single block only', 'Single + paired (J2)',
    'Without P-run windows', 'Without contact conditioning',
    'Without compact-core shaping', 'Without direct heat bath')))
COLORS = ('#167d9a', '#bd4858', '#765b9a', '#bc801b', '#38846c', '#67819d', '#4a4a4a')
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'axes.labelsize':9,
    'axes.titlesize':10, 'xtick.labelsize':8, 'ytick.labelsize':8,
    'pdf.fonttype':42, 'ps.fonttype':42, 'axes.spines.top':False,
    'axes.spines.right':False, 'legend.fontsize':8, 'savefig.facecolor':'white'})


def save_figure(fig, name):
    fig.savefig(ROOT / f'figures/{name}.pdf', bbox_inches='tight', pad_inches=.04)
    fig.savefig(ROOT / f'figures/{name}.png', dpi=500, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


def label(name):
    return name.replace('FCC','').replace('_', r'\_')


def bold(value):
    return r'\textbf{' + str(value) + '}'


def energy(value, native=None):
    if value is None:
        return '--'
    text = str(-value)
    return bold(text) if native is not None and value == native else text


def write_table(name, headings, rows, aligns=None):
    aligns = aligns or ('l' + 'r' * (len(headings)-1))
    text = [r'\begin{tabular}{@{}' + aligns + r'@{}}', r'\toprule',
            ' & '.join(headings) + r' \\', r'\midrule']
    for row in rows:
        text.append(' & '.join(str(cell) for cell in row) + r' \\')
    text.extend((r'\bottomrule', r'\end{tabular}'))
    (ROOT / f'tables/{name}.tex').write_text('\n'.join(text)+'\n', encoding='ascii')


def tables():
    rows = []
    for suite, names, native_source, lattice, hours, stop in (
        ('H1--H10','H1--H10',r'\cite{doduc2018,dotu2011}',8,1,'All workers'),
        ('F90',r'F90\_1--F90\_5',r'\cite{liu2013elp,liu2014iwl}',9,1,'All workers'),
        ('S','S1--S4',r'\cite{liu2013elp,liu2014iwl}',11,1,'All workers'),
        ('F180',r'F180\_1--F180\_3',r'\cite{liu2013elp,liu2014iwl}',12,3,'Disabled'),
        ('R','R1--R3',r'\cite{liu2013elp,liu2014iwl}',19,3,'Disabled'),
        ('CASP9-derived','3mse--3on7',r'\cite{liu2013elp,liu2014iwl}',r'12--19',6,'Disabled')):
        selected = [row for row in PRIMARY if row['suite'] == {'H1--H10':'H1-H10','S':'S1-S4','R':'R1-R3'}.get(suite,suite)]
        n = sorted(set(row['N'] for row in selected))
        ntext = str(n[0]) if len(n)==1 else f'{n[0]}--{n[-1]}'
        rows.append((suite,names,len(selected),ntext,lattice,hours,stop,native_source))
    write_table('benchmark_protocol', ('Suite','Instances','Count','$N$','$L$','$W$ (h)','Target stopping','Source'), rows,
                'llrrrrll')
    parameters = [
        ('Workers / base seed',r'16 / 21013001'),
        ('Worker seed stride',r'1000003'),
        ('Temperature endpoints',r'$8.0\rightarrow0.02$'),
        ('Requested cycle lengths',r'$80000\times2^s$ proposals'),
        ('Stagnant cycles for advancement','2 complete cycles'),
        ('Estimated cycle-duration cap',r'$W/8$'),
        ('Terminal-cooling trigger',r'Requested cycle does not fit, or $\leq5\%$ budget'),
        ('Ordinary block caps',r'$12,16,20,24$ residues'),
        ('Compact-core multipliers',r'$1.0,0.66,0.33,0.0$'),
        ('Pair / shaping coefficients',r'$-2.0$ / $-0.5$'),
        ('Single-block beam widths','24, 48, 96 (resource-controlled)'),
        ('Joint beam / expansion cap','24 / 12000 per proposal'),
        ('P-run / eligible conditioning',r'$0.50$ / $0.25$'),
        ('Archive capacity / contact gap','16 folds / 2 contacts'),
        ('P-refresh patience / cooldown','1200 s / 600 s'),
        ('P-refresh attempts / expansion cap','128 / 12000 per attempt'),
        ('Vacancy-guidance pulse','256000 ordinary proposals'),
        ('Kernel control chunk','8 proposals')]
    write_table('parameters', ('Parameter','Value'), parameters, 'll')
    hrows = []
    for row in PRIMARY[:10]:
        hrows.append((row['benchmark'],row['N'],energy(row['native_HH'],row['native_HH']),
                      energy(row['best_HH'],row['native_HH']),f'{row["mean_HH"]:.3f}',
                      f'{row["SD_HH"]:.3f}',f'{row["native_hits"]}/16',f'{row["first_best_s"]:.2f}'))
    write_table('h_results', ('Instance','$N$','Native $E$','CMHA $E$',r'$\overline C$','$s_C$','Hits','First (s)'),hrows)
    reference_file = ROOT / 'data/literature_contacts.json'
    literature = json.loads(reference_file.read_text()) if reference_file.exists() else {}
    rows = []
    for row in PRIMARY[10:]:
        ref = literature.get(row['benchmark'],{})
        native = row['native_HH']
        rows.append((label(row['benchmark']), row['N'],row['L'],energy(native,native),
            energy(ref.get('ELP_plus'),native),energy(row['IWL_HH'],native),energy(ref.get('K_ACO'),native),
            energy(row['best_HH'],native),fr'${row["mean_HH"]:.4f}\pm{row["SD_HH"]:.3f}$',
            f'{row["native_hits"]}/16' if native is not None else '--',
            f'{row["first_best_s"]/60:.2f}',int(row['wall_s']/3600)))
    write_table('literature_results', ('Instance','$N$','$L$','Native',r'ELP+\cite{liu2013elp}',
        r'IWL\cite{liu2014iwl}',r'K-ACO\cite{doduc2018}',r'\textbf{CMHA}',r'$\overline C\pm s_C$','Hits','First (min)','$W$ (h)'),rows)
    benches = ('F180_1FCC','R3FCC','3mr7','3no3')
    rows, threshold_rows = [], []
    for variant in ORDER:
        cells, thresholds = [LABELS[variant]], [LABELS[variant]]
        for benchmark in benches:
            row = next(r for r in ABLATIONS if r['variant']==variant and r['benchmark']==benchmark)
            maximum = max(r['mean_HH'] for r in ABLATIONS if r['benchmark']==benchmark)
            best = str(row['best_HH'])
            summary = fr'{row["mean_HH"]:.4f}\pm{row["SD_HH"]:.3f}'
            if abs(row['mean_HH']-maximum)<1e-10:
                best, summary = r'\mathbf{' + best + '}', r'\boldsymbol{' + summary + '}'
            cells.append(r'\shortstack[c]{$' + best + r'$\\$' + summary + '$}')
            minutes = f'{row["first_threshold_s"]/60:.2f}' if row['first_threshold_s'] is not None else 'NR'
            thresholds.append(f'{row["threshold_hits"]}/16; {minutes}')
        rows.append(cells)
        threshold_rows.append(thresholds)
    write_table('ablation_results', ('Configuration',r'F180\_1', 'R3','3mr7','3no3'), rows,'lcccc')
    write_table('ablation_thresholds', ('Configuration',r'F180\_1: 378','R3: 385','3mr7: 350','3no3: 474'), threshold_rows,'lcccc')
    paired_rows = []
    for row in PRIMARY:
        if row['suite'] in ('F180','R1-R3'):
            paired_rows.append((label(row['benchmark']),row['L'],row['previous_QABHA_HH'],row['best_HH'],
                row['best_HH']-row['previous_QABHA_HH']))
    write_table('development_comparison', ('Instance','$L$','Q-ABHA','CMHA',r'$\Delta C$'),paired_rows)
    begin_seq_table = r'\begin{tabular}{@{}lrr>{\raggedright\arraybackslash}p{0.73\textwidth}@{}}'
    seq_heading = r'\toprule Instance & $N$ & $L$ & HP sequence \\ \midrule'
    seq = [r'\begin{center}',r'\captionof{table}{Exact HP sequences of the 31 reported FCC instances. Spaces separate groups of ten residues only; removing spaces gives the original sequence.}\label{tab:sequences}',
           begin_seq_table, seq_heading]
    for i, row in enumerate(PRIMARY):
        if i == 15:
            seq.extend((r'\bottomrule',r'\end{tabular}',r'\end{center}',r'\clearpage',
                        r'\begin{center}',r'\textbf{Table~\ref{tab:sequences} (continued).}\par\medskip',
                        begin_seq_table,seq_heading))
        string = ' '.join(row['seq'][i:i+10] for i in range(0,len(row['seq']),10))
        seq.append(f'{label(row["benchmark"])} & {row["N"]} & {row["L"]} & '+r'{\ttfamily\footnotesize '+string+r'} \\[3pt]')
    seq.extend((r'\bottomrule',r'\end{tabular}',r'\end{center}'))
    (ROOT/'tables/benchmark_sequences.tex').write_text('\n'.join(seq)+'\n',encoding='ascii')
    macros = [r'\newcommand{\DataFreezeDate}{21 October 2026}']
    (ROOT/'tables/data_macros.tex').write_text('\n'.join(macros)+'\n',encoding='ascii')


def geometry():
    fig = plt.figure(figsize=(3.45,3.1))
    ax = fig.add_subplot(111,projection='3d')
    deltas = np.asarray([(a,b,0) for a in (-1,1) for b in (-1,1)] +
                        [(a,0,b) for a in (-1,1) for b in (-1,1)] +
                        [(0,a,b) for a in (-1,1) for b in (-1,1)])
    ax.add_collection3d(Line3DCollection([[np.zeros(3),point] for point in deltas],colors='#6c7783',linewidths=1.0))
    ax.scatter(*deltas.T,s=110,c='#2a91ad',edgecolors='white',linewidths=.6,depthshade=True)
    ax.scatter(0,0,0,s=170,c='#c94355',edgecolors='white',linewidths=.7,depthshade=True)
    ax.set(xlim=(-1.5,1.5),ylim=(-1.5,1.5),zlim=(-1.5,1.5),xlabel='$x$',ylabel='$y$',zlabel='$z$')
    ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1]);ax.set_zticks([-1,0,1])
    ax.set_box_aspect((1,1,1));ax.view_init(24,-57)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.fill=False;axis._axinfo['grid'].update(color='#dddddd',linewidth=.45)
    fig.subplots_adjust(left=0,right=.98,bottom=0,top=1)
    save_figure(fig,'fig01_fcc_geometry')


def multi_block():
    fig, axes = plt.subplots(2,1,figsize=(7.15,2.7),gridspec_kw={'height_ratios':[1,1.2]})
    intervals = ((4,10),(17,23),(29,35),(39,45))
    colors = ('#167d9a','#b54862','#4c8e62','#8d70a8')
    ax=axes[0]
    ax.plot([0,47],[0,0],c='#9ca5ad',lw=1.3,zorder=1)
    ax.scatter(np.arange(48),np.zeros(48),s=12,c='#ced3d7',zorder=2)
    for k,((a,b),color) in enumerate(zip(intervals,colors)):
        ax.scatter(np.arange(a,b),np.zeros(b-a),s=28,c=color,zorder=3)
        ax.scatter([a-1,b],[0,0],s=39,facecolors='white',edgecolors='#303840',lw=1,zorder=4)
        ax.text((a+b-1)/2,.24,f'Block {chr(65+k)}',ha='center',va='bottom',fontsize=9,color=color)
        ax.plot([a,b-1],[-.22,-.22],c=color,lw=2)
    ax.set_xlim(-1,49);ax.set_ylim(-.5,.7);ax.axis('off')
    ax.text(-.015,.96,'(a)',transform=ax.transAxes,fontweight='bold',fontsize=10)
    ax=axes[1]
    ax.set_xlim(0,10);ax.set_ylim(0,2);ax.axis('off')
    positions=[(.1,3.0,'Release selected intervals\nKeep exterior residues fixed'),
               (3.6,3.0,'Alternate extensions\nin one joint beam'),
               (7.1,2.8,'Score complete union\nDirect heat-bath selection')]
    for x,width,text in positions:
        ax.text(x+width/2,1.1,text,ha='center',va='center',fontsize=9,linespacing=1.5,
            bbox=dict(boxstyle='square,pad=.65',facecolor='#f1f5f7',edgecolor='#b5c5ce',lw=.8))
    for a,b in ((3.15,3.55),(6.65,7.05)):
        ax.add_patch(FancyArrowPatch((a,1.1),(b,1.1),arrowstyle='-|>',mutation_scale=13,lw=1,color='#465b67'))
    ax.text(-.015,.96,'(b)',transform=ax.transAxes,fontweight='bold',fontsize=10)
    fig.subplots_adjust(left=.03,right=.98,top=.95,bottom=.05,hspace=.15)
    save_figure(fig,'fig02_multiblock')


def convergence():
    fig,axes=plt.subplots(1,2,figsize=(7.15,2.85),layout='constrained')
    variants = [ORDER[index] for index in (0,1,2,6)]
    x=np.linspace(0,360,721)
    for ax,benchmark in zip(axes,('3mr7','3no3')):
        for variant in variants:
            row=next(r for r in ABLATIONS if r['variant']==variant and r['benchmark']==benchmark)
            curves=[]
            for worker in row['convergence']:
                events=worker['events']
                times=np.array([event[1]/60 for event in events])
                contacts=np.array([event[0] for event in events])
                indices=np.searchsorted(times,x,side='right')-1
                curves.append(contacts[np.maximum(indices,0)])
            values=np.array(curves)
            color=COLORS[ORDER.index(variant)]
            median=np.median(values,axis=0)
            lower,upper=np.percentile(values,[25,75],axis=0)
            ax.step(x,median,where='post',lw=1.5,color=color,label=LABELS[variant])
            ax.fill_between(x,lower,upper,step='post',color=color,alpha=.11,lw=0)
        reference={'3mr7':353,'3no3':477}[benchmark]
        ax.axhline(reference,c='#222222',lw=.9,ls='--')
        ax.text(354,reference+.15,'IWL',ha='right',va='bottom',fontsize=8)
        ax.set(title=benchmark,xlabel='Annealing wall time (min)',ylabel='Worker-best H-H contacts',xlim=(0,360))
        ax.set_ylim((334,354.5) if benchmark=='3mr7' else (450,479))
        ax.grid(axis='y',color='#dddddd',lw=.5)
        ax.set_xticks([0,60,120,180,240,300,360])
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False,fontsize=8)
    save_figure(fig,'fig03_convergence')


def fold_grid():
    names=('S3FCC','R3FCC','3no3','3on7')
    fig=plt.figure(figsize=(7.15,6.5))
    common_span = max(np.ptp(np.asarray(FOLDS[name]['coords']),axis=0).max() for name in names)+1.0
    for i,name in enumerate(names):
        item=FOLDS[name]
        coords=np.asarray(item['coords'],dtype=float)
        ax=fig.add_subplot(2,2,i+1,projection='3d')
        ax.set_proj_type('ortho')
        segments=np.stack([coords[:-1],coords[1:]],axis=1)
        ax.add_collection3d(Line3DCollection(segments,colors='#647584',linewidths=.85,alpha=.83))
        hydrophobic=np.array([residue=='H' for residue in item['seq']])
        for mask,color,size in ((~hydrophobic,'#6bb8cb',11),(hydrophobic,'#c54258',23)):
            ax.scatter(*coords[mask].T,s=size,c=color,edgecolors='white',linewidths=.18,depthshade=True,alpha=.97)
        side=item['L']
        center=(coords.min(axis=0)+coords.max(axis=0))/2
        bounds=[(value-common_span/2,value+common_span/2) for value in center]
        ax.set(xlim=bounds[0],ylim=bounds[1],zlim=bounds[2],xlabel='$x$',ylabel='$y$',zlabel='$z$')
        for axis,(lo,hi) in zip((ax.xaxis,ax.yaxis,ax.zaxis),bounds):
            axis.set_ticks(np.linspace(np.ceil(lo),np.floor(hi),3).astype(int))
        ax.tick_params(labelsize=7,pad=0)
        ax.set_box_aspect((1,1,1));ax.view_init(24,-58)
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            axis.pane.fill=False;axis._axinfo['grid'].update(color='#dddddd',linewidth=.4)
        text=f'({chr(97+i)}) {name.replace("FCC","").replace("_"," ")}: N={len(item["seq"])}, C={item["HH"]}, L={side}'
        ax.set_title(text,fontsize=9,pad=1)
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#c54258',label='H residue'),
                        Line2D([],[],marker='o',ls='',color='#6bb8cb',label='P residue')],
               loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.005),fontsize=9)
    fig.subplots_adjust(left=0,right=.98,bottom=.065,top=.98,hspace=.12,wspace=.02)
    save_figure(fig,'fig04_validated_folds')


def main():
    (ROOT/'figures').mkdir(exist_ok=True)
    (ROOT/'tables').mkdir(exist_ok=True)
    tables();geometry();multi_block();convergence();fold_grid()
    print('Built tables and four vector/raster manuscript figures from frozen data.')


if __name__=='__main__':
    main()
