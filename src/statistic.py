from turtle import st
import pandas as pd
from scipy import stats
import numpy as np

def read_data(results_file, sheet_name="Statistic"):
    return pd.read_excel(results_file, sheet_name=sheet_name)

def ttest_analysis(pd_data, fir_col="FCN", sec_col="FCN-CSA"):
    s_value, p_value = stats.ttest_ind(pd_data[fir_col], pd_data[sec_col])
    print(s_value)
    print(p_value)

def chi_squared_analysis(pd_data, fir_col="FCN", sec_col="FCN-CSA", count_col="Number-of-incidents"):
    count = 0
    total = 0
    diff = []
    for i in range(len(pd_data)):
        n1 = pd_data[fir_col][i]
        n2 = pd_data[sec_col][i]
        try:
            n1 = float(n1)
            n2 = float(n2)
        except:
            continue
        total = total + 1
        n = pd_data[count_col][i]
        # p_value = stats.chisquare([n1 * n, n2 * n], [(n1+n2) * n * 0.5, (n1+n2) * n * 0.5]).pvalue
        survivors = np.array([[n1 * n, n2 * n], [(1-n1) * n, (1-n2) * n]])
        p_value = stats.chi2_contingency(survivors)[1]
        if p_value < 0.05:
            count = count + 1
            diff.append(n2 - n1)
            diff.append(n2)
            diff.append(n1)

        # print(n1, n2)
        # print(p_value)
        # p_value = stats.chisquare([553, 547], [1100 * 0.5, 1100 * 0.5])
        # print(p_value)
        # print("====")
    print(count, total, diff)

    # fcn_accu <- data.frame(dataset = accu_df$Datasets)

    # fcn_accu$fcn_wo <- accu_df$`FCN-w-o`
    # fcn_accu$fcn_w <- accu_df$`FCN-w`
    # fcn_accu$fcn_n1 <- accu_df$`FCN-w-o`*accu_df$`Number-of-incidents`
    # fcn_accu$fcn_n2 <- accu_df$`FCN-w`*accu_df$`Number-of-incidents`
    # fcn_accu$p_val <- rep(1,28)
    # for (i in 1:28){
    #     n1 <- fcn_accu$fcn_n1[i]
    #     n2 <- fcn_accu$fcn_n2[i]
    #     n <- accu_df$`Number-of-incidents`[i]
    #     fcn_accu$p_val[i] <- round(as.numeric(prop.test(c(n1,n2),c(n,n))$p.value),3)
    # }




if __name__ == '__main__':
    results_file = "./log_final.xlsx"
    # pd_data = read_data(results_file)
    # # ttest_analysis(pd_data, "FCN", "FCN-CSA")
    # # ttest_analysis(pd_data, "FCN-CSA", "FCN")
    # # ttest_analysis(pd_data, "MLSTM", "MLSTM-CSA")
    # # ttest_analysis(pd_data, "MLSTM-FCN", "MLSTM-FCN-CSA")
    # # ttest_analysis(pd_data, "TAPNET", "TAPNET-CSA")
    # chi_squared_analysis(pd_data, "FCN", "FCN-CSA")
    # chi_squared_analysis(pd_data, "MLSTM", "MLSTM-CSA")
    # chi_squared_analysis(pd_data, "MLSTM-FCN", "MLSTM-FCN-CSA")
    # chi_squared_analysis(pd_data, "TAPNET", "TAPNET-CSA")
    # chi_squared_analysis(pd_data, "CNN", "CNN-CSA")

    pd_data = read_data(results_file, "Statistic_UTS")
    # ttest_analysis(pd_data, "FCN", "FCN-CSA")
    # ttest_analysis(pd_data, "FCN-CSA", "FCN")
    # ttest_analysis(pd_data, "MLSTM", "MLSTM-CSA")
    # ttest_analysis(pd_data, "MLSTM-FCN", "MLSTM-FCN-CSA")
    # ttest_analysis(pd_data, "TAPNET", "TAPNET-CSA")
    chi_squared_analysis(pd_data, "FCN", "FCN-CSA")
    # chi_squared_analysis(pd_data, "MLSTM", "MLSTM-CSA")
    # chi_squared_analysis(pd_data, "MLSTM-FCN", "MLSTM-FCN-CSA")
    # chi_squared_analysis(pd_data, "TAPNET", "TAPNET-CSA")
    # chi_squared_analysis(pd_data, "CNN", "CNN-CSA")


    pd_data = read_data(results_file, "Statistic_DTW")
    chi_squared_analysis(pd_data, "DTW", "MLSTM-FCN-CSA")


    

