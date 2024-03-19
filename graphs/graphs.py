import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import scienceplots

def quackml_vs_sklearn():
    style.use(['science','no-latex'])
    colors =  ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

    sklearn_times = [
        -5.1263035725,
        -6.5557615619,
        -6.8602088578,
        -6.8260886790,
        -6.4481115329,
        -6.7307637199,
        -6.4203305040,
        -5.9486423441,
        -5.3624977502,
        -4.3645369355,
        -3.9876373487,
        -2.8125778469,
        -1.9794309870,
        -1.1013410481,
        -0.1434213567,
        0.8140844603,
        1.4649755548,
        2.2393498559,
        3.2579922317
    ]

    sklearn_import = [ 
        0.0145652294,
        0.0030877590,
        0.0047590733,
        0.0062921047,
        0.0062570572,
        0.0070343018,
        0.0090548992,
        0.0136661530,
        0.0206668377,
        0.0452799797,
        0.0588490963,
        0.1374199390,
        0.2467858791,
        0.4547057152,
        0.8846178055,
        1.7193460464,
        2.6837968826,
        4.5655751228,
        9.1876180172
    ]

    sklearn_clean = [
        0.0037040710,
        0.0014619827,
        0.0024037361,
        0.0032508373,
        0.0102329254,
        0.0109910965,
        0.0192768574,
        0.0371501446,
        0.0769281387,
        0.1755230427,
        0.3269679546,
        0.6482150555,
        1.3546082973,
        2.3115918636,
        4.6642258167,
        9.2546870708,
        18.6161510944,
        37.3210041523,
        79.6060149670
    ]

    sklearn_train = [
        0.0140652657,
        0.0075418949,
        0.0038483143,
        0.0025212765,
        0.0051960945,
        0.0023810863,
        0.0026209354,
        0.0025250912,
        0.0036399364,
        0.0032649040,
        0.0041887760,
        0.0049209595,
        0.0068039894,
        0.0113773346,
        0.0207517147,
        0.0388360023,
        0.0767910480,
        0.1562671661,
        0.3788888454
    ]

    quackml_times = [
        -3.7973428156,
        -3.8798498121,
        -3.8057717119,
        -3.8371403170,
        -3.7416861124,
        -3.0595430479,
        -3.3453591601,
        -3.4578945248,
        -3.4514094960,
        -3.2016557903,
        -3.0593025106,
        -2.7470321725,
        -1.9802781619,
        -1.3677801854,
        -0.5289540842,
        0.3961255042,
        0.4145901945,
        0.5513339005,
        1.9064665733
    ]

    quackml_update = [
        0.0000830000,
        0.0001810000,
        0.0001840000,
        0.0003570000,
        0.0006350000,
        0.0018490000,
        0.0045120000,
        0.0054950000,
        0.0120200000,
        0.0317200000,
        0.0478030000,
        0.0813830000,
        0.1751850000,
        0.3208800000,
        0.6252930000,
        1.2471420000,
        1.2579660000,
        1.3997920000,
        3.6733970000
    ]

    quackml_finalise = [
        0.0718430000,
        0.0677470000,
        0.0713230000,
        0.0696120000,
        0.0741200000,
        0.1180970000,
        0.0938770000,
        0.0855110000,
        0.0793960000,
        0.0769740000,
        0.0721630000,
        0.0675740000,
        0.0782560000,
        0.0666070000,
        0.0677640000,
        0.0688270000,
        0.0749540000,
        0.0656480000,
        0.0755010000
    ]

    dataset_sizes = [i for i in range(2, 21)]

    # plt.plot(dataset_sizes, sklearn_times, marker='o', label='$\mathrm{DuckDB\ +\ sklearn}$')
    # plt.plot(dataset_sizes, quackml_times, marker='o', label='$\mathrm{QuackML}$')

    # # Add labels and title
    # plt.xlabel('$\log_2(|D|)$')
    # plt.xticks(np.arange(0, 22, 2))
    # plt.ylabel('$\log_2 \mathrm{\ model\ training\ time\ (s)}$')
    # plt.title('$\mathrm{Comparison\ of\ model\ training\ times\ between\ DuckDB\ +\ sklearn\ and\ QuackML}$')

    # # Set limits 
    # plt.xlim(0, 22)
    # plt.ylim(min(min(sklearn_times), min(quackml_times)) - 1, max(max(sklearn_times), max(quackml_times)) + 1)

    # Plot with 3 horisontal figures 
    fig, axs = plt.subplots(3, 1, figsize=(7, 8.5), gridspec_kw={'height_ratios': [1.8, 1, 1]})

    # Remove labels
    #plt.setp(axs[0].get_xticklabels(), visible=False)
    #plt.setp(axs[1].get_xticklabels(), visible=False)
    # Remove ticks
    #plt.setp(axs[0].get_xticklines(), visible=False)
    #plt.setp(axs[1].get_xticklines(), visible=False)
    plt.xticks(np.arange(2, 22, 2))
    plt.xlabel('$\log_2 (|D|)$')

    # Top plot =====================================================================
    axs[0].plot(dataset_sizes, sklearn_times, marker='o', label='DuckDB + sklearn', zorder=10, clip_on=False)
    axs[0].plot(dataset_sizes, quackml_times, marker='o', label='QuackML', zorder=10, clip_on=False)

    axs[0].set_title('Model training times')
    axs[0].set_ylabel('$\log_2$ model training time (s)')
    axs[0].set_xlim(2, 20)
    axs[0].set_xticks(np.arange(2, 22, 2))
    axs[0].legend(frameon=True, framealpha=.7, loc='upper left')
    # Highlight area to right of x=14
    axs[0].axvspan(14, 21, color='green', alpha=0.2, linewidth=0)

    # Middle plot ==================================================================
    total_times = [a+b for a, b in zip(quackml_finalise, quackml_update)]
    # Convert times to percentages 
    update_proportion = [(i/j) for i, j in zip(quackml_update, total_times)]
    finalise_proportion = [(i/j) for i, j in zip(quackml_finalise, total_times)]

    axs[1].stackplot(dataset_sizes, update_proportion, finalise_proportion, labels=['Update', 'Finalise'])

    axs[1].set_title('Function runtime breakdown - QuackML')
    axs[1].set_ylim(0, 1)
    axs[1].set_xlim(2, 20)
    axs[1].set_xticks(np.arange(2, 22, 2))
    axs[1].set_ylabel('Proportion of runtime')
    axs[1].legend(frameon=True, framealpha=.7, loc='upper left')

    # Vertical line and annotations
    axs[1].axvline(x=12.6, color='black', alpha = 0.5, linestyle='--')

    # Bottom plot ==================================================================
    total_times = [a+b for a, b in zip(sklearn_import, sklearn_train)]
    # Convert times to percentages
    import_proportion = [(i/j) for i, j in zip(sklearn_import, total_times)]
    train_proportion = [(i/j) for i, j in zip(sklearn_train, total_times)]

    axs[2].stackplot(dataset_sizes, import_proportion, train_proportion, labels=['Import', 'Train'])

    axs[2].set_title('Function runtime breakdown - DuckDB + sklearn')
    axs[2].set_ylim(0, 1)
    axs[2].set_xlim(2, 20)
    axs[2].set_xticks(np.arange(2, 22, 2))
    axs[2].set_ylabel('Proportion of runtime')
    axs[2].legend(frameon=True, framealpha=.7, loc = 'upper left')

    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.savefig('graphs/stacked_plot.png', dpi=300, bbox_inches='tight')

    plt.show()

def finalise_optimisations():
    style.use(['science','no-latex'])
    colors =  ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

    convergence_times = [
        314,
        1363,
        34545,
        71287,
        23393,
        17604,
        15508,
        19822,
        27748,
        45466,
        45783,
        55867,
        63059,
        63689,
        70357,
        61549,
        63424,
        59440,
        61790
    ]

    decay_times = [
        5814,
        23177,
        49235,
        44326,
        43443,
        25443,
        5091,
        2934,
        1986,
        1812,
        850,
        609,
        660,
        334,
        195,
        193,
        522,
        3213,
        8218
    ]

    initial_times = [
        71843,
        67747,
        71323,
        69612,
        74120,
        118097,
        93877,
        85511,
        79396,
        76974,
        72163,
        67574,
        78256,
        66607,
        67764,
        68827,
        74954,
        65648,
        75501
    ]

    dataset_sizes = [i for i in range(2, 21)]

    plt.figure(figsize=(9, 7))

    plt.plot(dataset_sizes, np.log2(initial_times), marker='o', label='Unoptimised runtime')
    plt.plot(dataset_sizes, np.log2(convergence_times), marker='o', label='Convergence testing')
    plt.plot(dataset_sizes, np.log2(decay_times), marker='o', label='Convergence testing and decay')
    plt.title('Finalise function runtimes')
    plt.xlabel('$\log_2 |D|$')
    plt.xlim(1, 21)
    plt.xticks(np.arange(2, 22, 2))
    plt.ylabel('$\log_2$ runtime ($\mu s$)')

    plt.legend(frameon=True, framealpha=.7, loc='lower left')
    plt.tight_layout()

    plt.savefig('graphs/finalise_optimisations.png', dpi=300, bbox_inches='tight')

def learning_rate_decay():
    style.use(['science','no-latex'])
    colors =  ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

    initial = 0.1 
    decay_step = 100

    plt.figure(figsize=(4, 4))

    iterations = np.arange(0, 5000, 1)
    plt.plot(iterations, initial * 0.90 ** (iterations / decay_step), label = '$r = 0.9$')
    plt.plot(iterations, initial * 0.925 ** (iterations / decay_step), label = '$r = 0.925$')
    plt.plot(iterations, initial * 0.95 ** (iterations / decay_step), label = '$r = 0.95$')
    plt.plot(iterations, initial * 0.99 ** (iterations / decay_step), label = '$r = 0.99$')
    plt.plot(iterations, initial * 0.999 ** (iterations / decay_step), label = '$r = 0.999$')
    plt.xlabel('Iterations')
    plt.ylabel('Learning rate')
    plt.title('Learning rate decay')
    plt.legend(frameon=True, framealpha=.7, loc='upper right')

    plt.tight_layout()

    plt.savefig('graphs/decay_rates.png', dpi=450, bbox_inches='tight')

    plt.show()

def quackml_fast_vs_sklearn():
    style.use(['science','no-latex'])
    colors =  ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

    duckdb_times = [
        0.0286304951,
        0.0106296539,
        0.0086073875,
        0.0088133812,
        0.0114531517,
        0.0094153881,
        0.0116758347,
        0.0161912441,
        0.0243067741,
        0.0485448837,
        0.0630378723,
        0.1423408985,
        0.2535898685,
        0.4660830498,
        0.9053695202,
        1.7581820488,
        2.7605879307,
        4.7218422890,
        9.5665068626
    ]

    quackml_slow_times = [
        0.0719260000,
        0.0679280000,
        0.0715070000,
        0.0699690000,
        0.0747550000,
        0.1199460000,
        0.0983890000,
        0.0910060000,
        0.0914160000,
        0.1086940000,
        0.1199660000,
        0.1489570000,
        0.2534410000,
        0.3874870000,
        0.6930570000,
        1.3159690000,
        1.3329200000,
        1.4654400000,
        3.7488980000
    ]

    quackml_update_times = [
        0.057403,
        0.051573,
        0.051258,
        0.051184,
        0.050985,
        0.05069,
        0.054863,
        0.056682,
        0.060681,
        0.073459,
        0.097454,
        0.145436,
        0.233158,
        0.419726,
        0.826874,
        1.513105,
        1.530512,
        2.283145,
        4.494879
    ]

    quackml_finalise_times = [
        0.005781,
        0.001928,
        0.007909,
        0.009716,
        0.068177,
        0.022022,
        0.012228,
        0.015387,
        0.02081,
        0.040285,
        0.062667,
        0.11895,
        0.230807,
        0.455497,
        0.942428,
        1.736466,
        1.786944,
        1.964633,
        4.320654
    ]

    quackml_optimised_times = [
        0.002543,
        0.003205,
        0.00669,
        0.011289,
        0.062509,
        0.019645,
        0.010582,
        0.011563,
        0.016742,
        0.028068,
        0.050832,
        0.093808,
        0.184403,
        0.411942,
        0.884638,
        1.619349,
        1.592769,
        1.892988,
        3.661965
    ]

    finalise_method = [
        0.000482,
        0.000784,
        0.005582,
        0.009403,
        0.060544,
        0.016951,
        0.006724,
        0.004578,
        0.00314,
        0.001717,
        0.001269,
        0.000849,
        0.000559,
        0.000614,
        0.000314,
        0.000242,
        0.000542,
        0.005355,
        0.020585
    ]

    update_method = [
        0.000092,
        0.000136,
        0.0002,
        0.00055,
        0.000829,
        0.001672,
        0.002926,
        0.005916,
        0.012519,
        0.024118,
        0.048007,
        0.090286,
        0.180735,
        0.40648,
        0.875044,
        1.619107,
        1.592227,
        1.887633,
        3.64138
    ]

    dataset_sizes = [i for i in range(2, 21)]

    fig, axs = plt.subplots(2, 1, figsize=(7.5, 8.5), gridspec_kw={'height_ratios': [1.8, 1]})

    plt.xticks(np.arange(2, 22, 2))
    plt.xlabel('$\log_2 (|D|)$')

    # Top plot =================================================================
    axs[0].plot(dataset_sizes, np.log2(duckdb_times), marker='o', label='DuckDB + sklearn', zorder=10, clip_on=False)
    axs[0].plot(dataset_sizes, np.log2(quackml_slow_times), marker='o', label='Unoptimised QuackML', zorder=10, clip_on=False)
    axs[0].plot(dataset_sizes, np.log2(quackml_update_times), marker='o', label='QuackML + optimised update', zorder=10, clip_on=False)
    axs[0].plot(dataset_sizes, np.log2(quackml_finalise_times), marker='o', label='QuackML + optimised finalise', zorder=10, clip_on=False)
    axs[0].plot(dataset_sizes, np.log2(quackml_optimised_times), marker='o', label='Optimised QuackML', zorder=10, clip_on=False)

    axs[0].set_title('Model training times')
    axs[0].set_ylabel('$\log_2$ model training time (s)')
    axs[0].set_xlim(2, 20)
    axs[0].set_xticks(np.arange(2, 22, 2))
    axs[0].legend(frameon=True, framealpha=.7, loc='upper left')
    # Highlight area to right of x=14
    axs[0].axvspan(8, 21, color='green', alpha=0.2, linewidth=0)    
    axs[0].axvspan(2, 4, color='green', alpha=0.2, linewidth=0)

    axs[1].axvline(x=8.8, color='black', alpha = 0.5, linestyle='--')

    # Bottom plot ==============================================================
    total_times = [a+b for a, b in zip(update_method, finalise_method)]
    update_proportion = [(i/j) for i, j in zip(update_method, total_times)]
    finalise_proportion = [(i/j) for i, j in zip(finalise_method, total_times)]

    axs[1].stackplot(dataset_sizes, update_proportion, finalise_proportion, labels=['Update', 'Finalise'])

    axs[1].set_title('Function runtime breakdown - QuackML')
    axs[1].set_ylim(0, 1)
    axs[1].set_xlim(2, 20)
    axs[1].set_xticks(np.arange(2, 22, 2))
    axs[1].set_ylabel('Proportion of runtime')
    axs[1].legend(frameon=True, framealpha=.7, loc='upper left') 

    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.savefig('graphs/quackml_fast_duckdb.png', dpi=300, bbox_inches='tight')

    plt.show()

def quackml_vs_cleaning_times():
    style.use(['science','no-latex'])
    colors =  ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

    quackml = [
        0.002543,
        0.003205,
        0.00669,
        0.011289,
        0.062509,
        0.019645,
        0.010582,
        0.011563,
        0.016742,
        0.028068,
        0.050832,
        0.093808,
        0.184403,
        0.411942,
        0.884638,
        1.619349,
        1.592769,
        1.892988,
        3.661965
    ]

    duckdb = [
        0.0323345661,
        0.0120916367,
        0.0110111237,
        0.0120642185,
        0.0216860771,
        0.0204064846,
        0.0309526920,
        0.0533413887,
        0.1012349129,
        0.2240679264,
        0.3900058270,
        0.7905559540,
        1.6081981659,
        2.7776749134,
        5.5695953369,
        11.0128691196,
        21.3767390251,
        42.0428464413,
        89.1725218296
    ]

    plt.figure(figsize=(6, 5))

    dataset_sizes = [i for i in range(2, 21)]
    plt.plot(dataset_sizes, np.log2(quackml), marker='o', label='QuackML', zorder=10, clip_on=False)
    plt.plot(dataset_sizes, np.log2(duckdb), marker='o', label='Duckdb + sklearn including\ndata processing times', zorder=10, clip_on=False)

    plt.legend(frameon=True, framealpha=.7, loc='upper left')
    plt.ylabel('$\log_2$ model training time (s)')
    plt.xlabel('$\log_2 (|D|)$')
    plt.title("Comparative model training times")

    plt.tight_layout()

    plt.xlim(2, 20)

    plt.savefig("graphs/sklearn_processing.png", dpi=300)
    plt.show()

def convergence_thresholds():
    dataset_sizes = [i for i in range(8, 21)]

    duckdb_times = [
        2.2879112493E-14,
        2.9899409796E-14,
        2.5802724878E-14,
        3.2604500467E-14,
        1.6595916784E-14,
        2.7777455172E-14,
        3.0004431193E-14,
        2.9882648140E-14,
        2.6746931676E-14,
        4.8790376978E-14,
        4.7136835774E-14,
        7.3413641440E-14,
        6.0053761954E-14
    ]

    thresh_1 = [
        1.2112500000E-02,
        5.8317200000E-03,
        2.8013900000E-03,
        1.3609600000E-03,
        6.2949500000E-04,
        2.4144700000E-04,
        1.5352600000E-04,
        6.1412200000E-05,
        1.1398400000E-05,
        3.7761200000E-06,
        6.1848300000E-06,
        4.1990400000E-06,
        1.1984000000E-06
    ]

    thresh_2 = [
        1.4210600000E-03,
        6.0643200000E-04,
        2.6360400000E-04,
        1.3119300000E-04,
        6.0329000000E-05,
        2.8741100000E-05,
        1.5292000000E-05,
        6.5811400000E-06,
        2.9254700000E-06,
        3.8905700000E-07,
        9.4796800000E-07,
        3.6799800000E-07,
        1.5338100000E-07
    ]

    thresh_3 = [
        1.2460100000E-04,
        6.3207300000E-05,
        2.5055900000E-05,
        1.2533500000E-05,
        5.6630600000E-06,
        2.5088000000E-06,
        1.0015900000E-06,
        7.1100200000E-07,
        1.9888400000E-07,
        4.0586100000E-08,
        7.5933700000E-08,
        3.9208100000E-08,
        1.9478900000E-08
    ]

    thresh_4 = [
        1.3501000000E-05,
        4.9981800000E-06,
        2.7596600000E-06,
        1.2167100000E-06,
        5.5576000000E-07,
        2.9673800000E-07,
        1.0523600000E-07,
        3.8635400000E-08,
        1.3498400000E-08,
        4.1987600000E-09,
        6.0160800000E-09,
        3.8475700000E-09,
        8.7244200000E-10
    ]

    thresh_5 = [
        1.3762200000E-06,
        5.9122700000E-07,
        2.7424600000E-07,
        1.3063400000E-07,
        5.7240000000E-08,
        2.7409200000E-08,
        1.1235500000E-08,
        4.3998600000E-09,
        3.5584300000E-09,
        4.3513900000E-10,
        8.8629700000E-10,
        3.5149000000E-10,
        1.0876900000E-10
    ]

    thresh_6 = [
        1.4955000000E-07,
        5.9824400000E-08,
        3.0080500000E-08,
        1.2096900000E-08,
        5.8777900000E-09,
        2.5191100000E-09,
        1.1914600000E-09,
        5.0469600000E-10,
        2.4888200000E-10,
        4.4907100000E-11,
        6.8139100000E-11,
        4.0451100000E-11,
        1.3485400000E-11
    ]

    thresh_7 = [
        1.4079900000E-08,
        6.1852300000E-09,
        2.6326700000E-09,
        1.2759800000E-09,
        6.4194100000E-10,
        3.1496600000E-10,
        1.3257700000E-10,
        5.7702300000E-11,
        1.7200000000E-11,
        4.9091800000E-12,
        5.0690500000E-12,
        4.4510300000E-12,
        1.6857400000E-12
    ]

    iters = [
        218.5, 236.8, 256.9, 278.4, 300.8, 324.5, 349.2
    ]

    style.use(['science','no-latex'])
    colors =  ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

    plt.figure(figsize=(10, 7))

    plt.plot(dataset_sizes, np.log(duckdb_times), marker='o', label='DuckDB + sklearn', zorder=10, clip_on=False)
    plt.plot(dataset_sizes, np.log(thresh_1), marker='o', label='Threshold = $10^{-1}$', zorder=10, clip_on=False)
    plt.plot(dataset_sizes, np.log(thresh_2), marker='o', label='Threshold = $10^{-2}$', zorder=10, clip_on=False)
    plt.plot(dataset_sizes, np.log(thresh_3), marker='o', label='Threshold = $10^{-3}$', zorder=10, clip_on=False)
    plt.plot(dataset_sizes, np.log(thresh_4), marker='o', label='Threshold = $10^{-4}$', zorder=10, clip_on=False)
    plt.plot(dataset_sizes, np.log(thresh_5), marker='o', label='Threshold = $10^{-5}$', zorder=10, clip_on=False)
    plt.plot(dataset_sizes, np.log(thresh_6), marker='o', label='Threshold = $10^{-6}$', zorder=10, clip_on=False)
    plt.plot(dataset_sizes, np.log(thresh_7), marker='o', label='Threshold = $10^{-7}$', zorder=10, clip_on=False)

    label_positions = [
        l[-1] for l in [thresh_1, thresh_2, thresh_3, thresh_4, thresh_5, thresh_6, thresh_7]
    ]

    for i in range(len(label_positions)):
        # plt.plot(
        #     [20, 20.5],
        #     [np.log(label_positions[i]), np.log(label_positions[i])],
        #     color = colors[i+1],
        #     linestyle = '--',
        #     alpha = .5
        # )

        plt.text(
            20.3,
            np.log(label_positions[i]),
            f'Iterations: {iters[i]}',
            color = colors[i+1],
            fontsize=15,
            weight='bold',
            va='center',
            zorder = 15
        )

    plt.xlabel('$\log_2 (|D|)$', fontsize=15)
    plt.xlim(8, 23)
    plt.ylim(-33, -3.5)
    plt.ylabel('$\log$ Root Mean Squared Error', fontsize=15)
    plt.title('Comparison of Model Variation Accuracies', fontsize=15)

    plt.tight_layout()
    plt.legend(frameon=True, framealpha=.7, loc='upper right')

    plt.savefig('graphs/thresholds.png', dpi=300, bbox_inches='tight')

    plt.show()

def join_performance():
    unintegrated_import = [
        0.235819101333618,
        0.017235279083252,
        0.0314950942993164,
        0.140176296234131,
        1.48333382606506,
        12.3582508563995,
        164.58486533165
    ]

    unintegrated_join = [
        0.000802040100097656,
        0.00193405151367188,
        0.0160148143768311,
        0.0217957496643066,
        0.224545001983643,
        1.23244619369507,
        11.9429259300232
    ]

    unintegrated_train = [
        0.00382208824157715,
        0.00376605987548828,
        0.0142452716827393,
        0.0051729679107666,
        0.0345890522003174,
        0.357449293136597,
        16.0980980396271
    ]

    factorised_lift = [
        6455,
        3513,
        6245,
        7425,
        9369,
        10981,
        7415,
        16427,
        8634,
        8700
    ]

    factorised_train = [
        4920,
        5087,
        7151,
        8166,
        24486,
        24807,
        81452,
        55543,
        133878,
        175768
    ]

    loose_join = [
        683,
        4414,
        11921,
        108533,
        1379771,
        15959325,
        94801702
    ]

    loose_train = [
        4325,
        6821,
        13354,
        87000,
        1060834,
        3879212,
        36739040
    ]

    unintegrated_totals = [(a+b+c) for a, b, c in zip(unintegrated_import, unintegrated_join, unintegrated_train)]
    factorised_totals = [(a+b)/1000000 for a, b in zip(factorised_lift, factorised_train)]
    loose_totals = [(a+b)/1000000 for a, b in zip(loose_join, loose_train)]
    
    xs = [i for i in range(1, 11)]

    style.use(['science','no-latex'])
    colors =  ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

    fig, axs = plt.subplots(4, 1, figsize=(7, 8.5), gridspec_kw={'height_ratios': [1.8, 1, 1, 1]})

    # Top plot - lines =========================================================
    axs[0].plot(xs[:7], np.log2(unintegrated_totals), marker='o', label='DuckDB + sklearn', zorder=10, clip_on=False)
    axs[0].plot(xs[:7], np.log2(loose_totals), marker='o', label='Loosely-integrated QuackML', zorder=10, clip_on=False)
    axs[0].plot(xs, np.log2(factorised_totals), marker='o', label='Factorised QuackML', zorder=10, clip_on=False)

    axs[0].set_title('Model training times over cartesian products of input relations')
    axs[0].set_ylabel('$\log_2$ model training time (s)')
    axs[0].legend(frameon=True, framealpha=.7, loc='upper left')
    axs[0].set_xlim(1, 10)
    axs[0].set_xticks(np.arange(1, 11, 1))

    # Second plot - stacked duckdb+sklearn =====================================
    import_proportion = [(i/j) for i, j in zip(unintegrated_import, unintegrated_totals)]
    join_proportion = [(i/j) for i, j in zip(unintegrated_join, unintegrated_totals)]
    train_proportion = [(i/j) for i, j in zip(unintegrated_train, unintegrated_totals)] 

    axs[1].stackplot(xs[:7], import_proportion, join_proportion, train_proportion, labels=['Import', 'Construct Join', 'Train Model'])
    axs[1].set_xlim(1, 10)
    axs[1].set_xticks(np.arange(1, 11, 1))
    axs[1].set_ylabel('Proportion of runtime')
    axs[1].set_title('Function runtime breakdown - DuckDB + sklearn')
    axs[1].set_ylim(0, 1)
    axs[1].legend(frameon=True, framealpha=.7, loc='upper right')

    # Third plot - stack unfactorised ========================================== 
    join_proportion = [(i/(j*1000000)) for i, j in zip(loose_join, loose_totals)]
    train_proportion = [(i/(j*1000000)) for i, j in zip(loose_train, loose_totals)]

    axs[2].stackplot(xs[:7], join_proportion, train_proportion, labels=['Construct Join', 'Train Model'])
    axs[2].set_xlim(1, 10)
    axs[2].set_xticks(np.arange(1, 11, 1))
    axs[2].set_ylabel('Proportion of runtime')
    axs[2].set_title('Function runtime breakdown - Loosely-integrated QuackML')
    axs[2].set_ylim(0, 1)
    axs[2].legend(frameon=True, framealpha=.7, loc='upper right')

    # Fourth plot - stack factorised ===========================================
    lift_proportion = [(i/(j*1000000)) for i, j in zip(factorised_lift, factorised_totals)]
    train_proportion = [(i/(j*1000000)) for i, j in zip(factorised_train, factorised_totals)]

    axs[3].stackplot(xs, lift_proportion, train_proportion, labels=['Lifting Function', 'Train Model'])
    axs[3].set_xlim(1, 10)
    axs[3].set_xticks(np.arange(1, 11, 1))
    axs[3].set_ylabel('Proportion of runtime')
    axs[3].set_title('Function runtime breakdown - Factorised QuackML')
    axs[3].set_ylim(0, 1)
    axs[3].legend(frameon=True, framealpha=.7, loc='upper right')
    axs[3].set_xlabel('Number of input relations')

    plt.tight_layout()
    plt.savefig('graphs/join_performance.png', dpi=300, bbox_inches='tight')

def join_performance_2():
    unintegrated_import = [
        0.00367116928100586,
        0.0061643123626709,
        0.00914692878723145,
        0.0125889778137207,
        0.0249152183532715,
        0.102702617645264,
        0.5037841796875,
        1.41124486923218,
        3.85844254493713,
        15.9945795536041
    ]

    unintegrated_join = [
        0.000869989395141602,
        0.002838134765625,
        0.00334024429321289,
        0.00444316864013672,
        0.00633716583251953,
        0.0201318264007568,
        0.115653038024902,
        0.364930152893066,
        0.48867392539978,
        2.02883005142212
    ]

    unintegrated_train = [
        0.00319504737854004,
        0.00331783294677734,
        0.00248908996582031,
        0.00265097618103027,
        0.00546598434448242,
        0.0123250484466553,
        0.0517146587371826,
        0.174834251403809,
        0.67720103263855,
        5.56078910827637
    ]

    factorised_lift = [
        4326,
        5022,
        5856,
        5998,
        7254,
        7263,
        9609,
        10969,
        11530,
        11324
    ]

    factorised_train = [
        1609,
        4665,
        9645,
        38274,
        64985,
        82316,
        137453,
        164482,
        219936,
        318577
    ]

    loose_join = [
        1609,
        5665,
        9645,
        48274,
        84985,
        102316,
        137453,
        164482,
        219936,
        318577
    ]

    loose_train = [
        2988,
        15071,
        27106,
        67286,
        133754,
        199382,
        432967,
        1294080,
        2615436,
        6398412
    ]

    unintegrated_totals = [(a+b+c) for a, b, c in zip(unintegrated_import, unintegrated_join, unintegrated_train)]
    factorised_totals = [(a+b)/1000000 for a, b in zip(factorised_lift, factorised_train)]
    loose_totals = [(a+b)/1000000 for a, b in zip(loose_join, loose_train)]
    
    xs = [i for i in range(1, 11)]

    style.use(['science','no-latex'])
    colors =  ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

    fig, axs = plt.subplots(4, 1, figsize=(7, 8.5), gridspec_kw={'height_ratios': [1.8, 1, 1, 1]})

    # Top plot - lines =========================================================
    axs[0].plot(xs, np.log2(unintegrated_totals), marker='o', label='DuckDB + sklearn', zorder=10, clip_on=False)
    axs[0].plot(xs, np.log2(loose_totals), marker='o', label='Loosely-integrated QuackML', zorder=10, clip_on=False)
    axs[0].plot(xs, np.log2(factorised_totals), marker='o', label='Factorised QuackML', zorder=10, clip_on=False)

    axs[0].set_title('Model training times over natual join of input relations')
    axs[0].set_ylabel('$\log_2$ model training time (s)')
    axs[0].legend(frameon=True, framealpha=.7, loc='upper left')
    axs[0].set_xlim(1, 10)
    axs[0].set_xticks(np.arange(1, 11, 1))

    axs[0].axvspan(1, 3.5, color='green', alpha=0.2, linewidth=0)
    axs[0].axvspan(5.5, 10, color='green', alpha=0.2, linewidth=0)

    # Second plot - stacked duckdb+sklearn =====================================
    import_proportion = [(i/j) for i, j in zip(unintegrated_import, unintegrated_totals)]
    join_proportion = [(i/j) for i, j in zip(unintegrated_join, unintegrated_totals)]
    train_proportion = [(i/j) for i, j in zip(unintegrated_train, unintegrated_totals)] 

    axs[1].stackplot(xs, import_proportion, join_proportion, train_proportion, labels=['Import', 'Construct Join', 'Train Model'])
    axs[1].set_xlim(1, 10)
    axs[1].set_xticks(np.arange(1, 11, 1))
    axs[1].set_ylabel('Proportion of runtime')
    axs[1].set_title('Function runtime breakdown - DuckDB + sklearn')
    axs[1].set_ylim(0, 1)
    axs[1].legend(frameon=True, framealpha=.7, loc='upper right')

    # Third plot - stack unfactorised ========================================== 
    join_proportion = [(i/(j*1000000)) for i, j in zip(loose_join, loose_totals)]
    train_proportion = [(i/(j*1000000)) for i, j in zip(loose_train, loose_totals)]

    axs[2].stackplot(xs, join_proportion, train_proportion, labels=['Construct Join', 'Train Model'])
    axs[2].set_xlim(1, 10)
    axs[2].set_xticks(np.arange(1, 11, 1))
    axs[2].set_ylabel('Proportion of runtime')
    axs[2].set_title('Function runtime breakdown - Loosely-integrated QuackML')
    axs[2].set_ylim(0, 1)
    axs[2].legend(frameon=True, framealpha=.7, loc='upper right')

    # Fourth plot - stack factorised ===========================================
    lift_proportion = [(i/(j*1000000)) for i, j in zip(factorised_lift, factorised_totals)]
    train_proportion = [(i/(j*1000000)) for i, j in zip(factorised_train, factorised_totals)]

    axs[3].stackplot(xs, lift_proportion, train_proportion, labels=['Lifting Function', 'Train Model'])
    axs[3].set_xlim(1, 10)
    axs[3].set_xticks(np.arange(1, 11, 1))
    axs[3].set_ylabel('Proportion of runtime')
    axs[3].set_title('Function runtime breakdown - Factorised QuackML')
    axs[3].set_ylim(0, 1)
    axs[3].legend(frameon=True, framealpha=.7, loc='upper right')
    axs[3].set_xlabel('Number of input relations')

    plt.tight_layout()
    plt.savefig('graphs/natural_join_performance.png', dpi=300, bbox_inches='tight')



if __name__ == '__main__':
    #quackml_vs_sklearn()
    #finalise_optimisations()
    #learning_rate_decay()
    #quackml_fast_vs_sklearn()
    #quackml_vs_cleaning_times()
    #convergence_thresholds()
    join_performance_2()