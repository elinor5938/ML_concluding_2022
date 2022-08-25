
import pandas as pd
import scikit_posthocs as sp
from pingouin import friedman

if __name__ == "__main__":
    main_path = "/Users/esraan/Desktop/ML_final/data_report_final.csv"  # "" /sise/home/esraan/ML_4/input/


    #testing AUC among the diffrenet alogrithms
    group_mrmr = pd.DataFrame(None)
    group_reliefF = pd.DataFrame(None)
    group_SelectFdr = pd.DataFrame(None)
    group_RFE = pd.DataFrame(None)
    group_hybrid_DReductionC = pd.DataFrame(None)
    group_GBC = pd.DataFrame(None)
    group_GBC_new = pd.DataFrame(None)
    df = pd.read_csv(main_path, low_memory=False)
    group_mrmr = pd.concat([group_mrmr,df.loc[df['Filtering Algorithm'].str.contains("mrmr", na=False)].filter(regex='^mean_',axis=1).filter(regex='AUC$',axis=1)],axis=0)
    group_reliefF = pd.concat([group_reliefF,df.loc[df['Filtering Algorithm'].str.contains("reliefF", na=False) ].filter(regex='^mean_',axis=1).filter(regex='AUC$',axis=1)],axis=0)
    group_SelectFdr = pd.concat([group_SelectFdr,df.loc[df['Filtering Algorithm'].str.contains("SelectFdr", na=False) ].filter(regex='^mean_',axis=1).filter(regex='AUC$',axis=1)],axis=0)
    group_RFE =pd.concat([group_RFE,df.loc[df['Filtering Algorithm'].str.contains("RFE", na=False) ].filter(regex='^mean_',axis=1).filter(regex='AUC$',axis=1)],axis=0)
    group_hybrid_DReductionC = pd.concat([group_hybrid_DReductionC,df.loc[df['Filtering Algorithm'].str.contains("DReductionC",na=False) ].filter(regex='^mean_',axis=1).filter(regex='AUC$',axis=1)],axis=0)
    group_GBC = pd.concat([group_GBC,df.loc[df['Filtering Algorithm'].str.contains("^GBC$", regex = True,na=False) ].filter(regex='^mean_',axis=1).filter(regex='AUC$',axis=1)],axis=0)
    group_GBC_new = pd.concat([group_GBC_new,df.loc[df['Filtering Algorithm'].str.contains("GBC_new",na=False) ].filter(regex='^mean_',axis=1).filter(regex='AUC$',axis=1)],axis=0)
    allgroups = pd.concat([group_mrmr.reset_index(drop=True), group_reliefF.reset_index(drop=True)], axis=1,
                          ignore_index=True)
    allgroups = pd.concat([allgroups, group_SelectFdr.reset_index(drop=True)], axis=1, ignore_index=True)
    allgroups = pd.concat([allgroups, group_RFE.reset_index(drop=True)], axis=1, ignore_index=True)
    allgroups = pd.concat([allgroups, group_hybrid_DReductionC.reset_index(drop=True)], axis=1, ignore_index=True)
    allgroups = pd.concat([allgroups, group_GBC.reset_index(drop=True)], axis=1, ignore_index=True)
    allgroups = pd.concat([allgroups, group_GBC_new.reset_index(drop=True)], axis=1, ignore_index=True)
    allgroups["id"] = range(1,len(allgroups)+1)
    allgroups.columns= ['mrmr','reliefF','SelectFdr','RFE','hybrid_DReductionC','GBC','GBC_new','id']
    new_all = pd.melt(allgroups,id_vars='id',var_name='alogrithm',value_name='AUC')

    result_stats = friedman(data= new_all,dv ='AUC',within='alogrithm',subject='id')
    if result_stats.iloc[:,4].values < 0.05: #0.05
        pv = sp.posthoc_conover(new_all, val_col='AUC', group_col='alogrithm')
        pv.columns = ['mrmr','reliefF','SelectFdr','RFE','hybrid_DReductionC','GBC','GBC_new']
        pv.index = ['mrmr','reliefF','SelectFdr','RFE','hybrid_DReductionC','GBC','GBC_new']

        pv.to_csv("/Users/esraan/Desktop/ML_final/post-hoc_AUC.csv")
    result_stats.to_csv("/Users/esraan/Desktop/ML_final/freidman_test_AUC.csv")

    # testing clf fit time among the diffrenet alogrithms
    allgroups =pd.DataFrame(None)
    group_mrmr =pd.DataFrame(None)
    group_reliefF = pd.DataFrame(None)
    group_SelectFdr = pd.DataFrame(None)
    group_RFE = pd.DataFrame(None)
    group_hybrid_DReductionC = pd.DataFrame(None)
    group_GBC = pd.DataFrame(None)
    group_GBC_new = pd.DataFrame(None)
    group_mrmr = pd.concat([group_mrmr,df.loc[df['Filtering Algorithm'].str.contains("mrmr", na=False)].filter(regex='^mean_fit',axis=1).filter(regex='time$',axis=1)],axis=0)
    group_reliefF = pd.concat([group_reliefF,df.loc[df['Filtering Algorithm'].str.contains("reliefF", na=False) ].filter(regex='^mean_fit',axis=1).filter(regex='time$',axis=1)],axis=0)
    group_SelectFdr = pd.concat([group_SelectFdr,df.loc[df['Filtering Algorithm'].str.contains("SelectFdr", na=False) ].filter(regex='^mean_fit',axis=1).filter(regex='time$',axis=1)],axis=0)
    group_RFE =pd.concat([group_RFE,df.loc[df['Filtering Algorithm'].str.contains("RFE", na=False) ].filter(regex='^mean_fit',axis=1).filter(regex='time$',axis=1)],axis=0)
    group_hybrid_DReductionC = pd.concat([group_hybrid_DReductionC,df.loc[df['Filtering Algorithm'].str.contains("DReductionC",na=False) ].filter(regex='^mean_fit',axis=1).filter(regex='time$',axis=1)],axis=0)
    group_GBC = pd.concat([group_GBC,df.loc[df['Filtering Algorithm'].str.contains("^GBC$", regex = True,na=False) ].filter(regex='^mean_fit',axis=1).filter(regex='time$',axis=1)],axis=0)
    group_GBC_new = pd.concat([group_GBC_new,df.loc[df['Filtering Algorithm'].str.contains("GBC_new",na=False) ].filter(regex='^mean_fit',axis=1).filter(regex='time$',axis=1)],axis=0)
    allgroups = pd.concat([group_mrmr.reset_index(drop=True), group_reliefF.reset_index(drop=True)], axis=1,
                          ignore_index=True)
    allgroups = pd.concat([allgroups, group_SelectFdr.reset_index(drop=True)], axis=1, ignore_index=True)
    allgroups = pd.concat([allgroups, group_RFE.reset_index(drop=True)], axis=1, ignore_index=True)
    allgroups = pd.concat([allgroups, group_hybrid_DReductionC.reset_index(drop=True)], axis=1, ignore_index=True)
    allgroups = pd.concat([allgroups, group_GBC.reset_index(drop=True)], axis=1, ignore_index=True)
    allgroups = pd.concat([allgroups, group_GBC_new.reset_index(drop=True)], axis=1, ignore_index=True)
    allgroups["id"] = range(1,len(allgroups)+1)
    allgroups.columns= ['mrmr','reliefF','SelectFdr','RFE','hybrid_DReductionC','GBC','GBC_new','id']
    new_all = pd.melt(allgroups,id_vars='id',var_name='alogrithm',value_name='AUC')

    result_stats = friedman(data= new_all,dv ='AUC',within='alogrithm',subject='id')
    if result_stats.iloc[:,4].values < 0.05: #0.05
        pv = sp.posthoc_conover(new_all, val_col='AUC', group_col='alogrithm')
        pv.columns = ['mrmr','reliefF','SelectFdr','RFE','hybrid_DReductionC','GBC','GBC_new']
        pv.index = ['mrmr','reliefF','SelectFdr','RFE','hybrid_DReductionC','GBC','GBC_new']

        pv.to_csv("/Users/esraan/Desktop/ML_final/post-hoc_clf_time.csv")
    result_stats.to_csv("/Users/esraan/Desktop/ML_final/freidman_test_clf_time.csv")

    # testing Fs time among the diffrenet alogrithms
    allgroups = pd.DataFrame(None)
    group_mrmr = pd.DataFrame(None)
    group_reliefF = pd.DataFrame(None)
    group_SelectFdr = pd.DataFrame(None)
    group_RFE = pd.DataFrame(None)
    group_hybrid_DReductionC = pd.DataFrame(None)
    group_GBC = pd.DataFrame(None)
    group_GBC_new = pd.DataFrame(None)
    group_mrmr = pd.concat([group_mrmr,df.loc[df['Filtering Algorithm'].str.contains("mrmr", na=False)].filter(regex='^FS',axis=1).filter(regex='time$',axis=1)],axis=0)
    group_reliefF = pd.concat([group_reliefF,df.loc[df['Filtering Algorithm'].str.contains("reliefF", na=False) ].filter(regex='^FS',axis=1).filter(regex='time$',axis=1)],axis=0)
    group_SelectFdr = pd.concat([group_SelectFdr,df.loc[df['Filtering Algorithm'].str.contains("SelectFdr", na=False) ].filter(regex='^FS',axis=1).filter(regex='time$',axis=1)],axis=0)
    group_RFE =pd.concat([group_RFE,df.loc[df['Filtering Algorithm'].str.contains("RFE", na=False) ].filter(regex='^FS',axis=1).filter(regex='time$',axis=1)],axis=0)
    group_hybrid_DReductionC = pd.concat([group_hybrid_DReductionC,df.loc[df['Filtering Algorithm'].str.contains("DReductionC",na=False) ].filter(regex='^FS',axis=1).filter(regex='time$',axis=1)],axis=0)
    group_GBC = pd.concat([group_GBC,df.loc[df['Filtering Algorithm'].str.contains("^GBC$", regex = True,na=False) ].filter(regex='^FS',axis=1).filter(regex='time$',axis=1)],axis=0)
    group_GBC_new = pd.concat([group_GBC_new,df.loc[df['Filtering Algorithm'].str.contains("GBC_new",na=False) ].filter(regex='^FS',axis=1).filter(regex='time$',axis=1)],axis=0)
    allgroups = pd.concat([group_mrmr.reset_index(drop=True), group_reliefF.reset_index(drop=True)], axis=1,
                          ignore_index=True)
    allgroups = pd.concat([allgroups, group_SelectFdr.reset_index(drop=True)], axis=1, ignore_index=True)
    allgroups = pd.concat([allgroups, group_RFE.reset_index(drop=True)], axis=1, ignore_index=True)
    allgroups = pd.concat([allgroups, group_hybrid_DReductionC.reset_index(drop=True)], axis=1, ignore_index=True)
    allgroups = pd.concat([allgroups, group_GBC.reset_index(drop=True)], axis=1, ignore_index=True)
    allgroups = pd.concat([allgroups, group_GBC_new.reset_index(drop=True)], axis=1, ignore_index=True)
    allgroups["id"] = range(1,len(allgroups)+1)
    allgroups.columns= ['mrmr','reliefF','SelectFdr','RFE','hybrid_DReductionC','GBC','GBC_new','id']
    new_all = pd.melt(allgroups,id_vars='id',var_name='alogrithm',value_name='AUC')
    result_stats = friedman(data= new_all,dv ='AUC',within='alogrithm',subject='id')
    if result_stats.iloc[:,4].values < 0.05:
        pv = sp.posthoc_conover(new_all, val_col='AUC', group_col='alogrithm')
        pv.columns = ['mrmr','reliefF','SelectFdr','RFE','hybrid_DReductionC','GBC','GBC_new']
        pv.index = ['mrmr','reliefF','SelectFdr','RFE','hybrid_DReductionC','GBC','GBC_new']

        pv.to_csv("/Users/esraan/Desktop/ML_final/post-hoc_FS_time.csv")
    # sns.heatmap(data=pv)
    result_stats.to_csv("/Users/esraan/Desktop/ML_final/freidman_test_FS_time.csv")



