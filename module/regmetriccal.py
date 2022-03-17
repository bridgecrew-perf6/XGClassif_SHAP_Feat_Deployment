def metric_cal(model=None, data=None, target=None):
    '''
    A function for calculating the metrics such as accuracy_score, recall_score, precision_score, f1_score, fbeta_score,
    roc_auc_score, cohen_kappa_score, hinge_loss, hamming_loss, log_loss using a model, and returning a dictionary that
    can be converted into a pandas dataframe
    '''
    import numpy as np
    from sklearn.metrics import (r2_score, mean_absolute_error,
                                 median_absolute_error, mean_squared_error)

    feat_sel = [fclass_feat, freg_feat, selk_class_feat, selk_reg_feat, sfm_dt_feat, sfm_rf_feat,
                sfm_xgb_feat, sfm_log_feat, selp_class_feat, selp_reg_feat, vt_feat_50]

    fs_model_dict = {'fclass': list(),
                     'freg': list(),
                     'selk_class': list(),
                     'selk_reg': list(),
                     'sfm_dt': list(),
                     'sfm_rf': list(),
                     'sfm_xgb': list(),
                     'sfm_log': list(),
                     'selp_class': list(),
                     'selp_reg': list(),
                     'vt_50': list()
                     }

    # 1
    y_pred_fclass = model.fit(data[fclass_feat], target).predict(data[fclass_feat])
    rsquare = round(r2_score(target, y_pred_fclass), 2)
    mean_ae = round(mean_absolute_error(target, y_pred_fclass), 2)
    median_ae = round(median_absolute_error(target, y_pred_fclass), 2)
    mse = round(mean_squared_error(target, y_pred_fclass), 2)
    rmse = round(np.sqrt(mean_squared_error(target, y_pred_fclass)), 2)
    adj_rsquare = round(1-(((1-rsquare)*(len(data)-1))/(len(data)-len(data.columns)-1)), 2)
    fs_model_dict['fclass'].append(rsquare)
    fs_model_dict['fclass'].append(mean_ae)
    fs_model_dict['fclass'].append(median_ae)
    fs_model_dict['fclass'].append(mse)
    fs_model_dict['fclass'].append(rmse)
    fs_model_dict['fclass'].append(adj_rsquare)

    # 2
    y_pred_freg = model.fit(data[freg_feat], target).predict(data[freg_feat])
    rsquare = round(r2_score(target, y_pred_freg), 2)
    mean_ae = round(mean_absolute_error(target, y_pred_freg), 2)
    median_ae = round(median_absolute_error(target, y_pred_freg), 2)
    mse = round(mean_squared_error(target, y_pred_freg), 2)
    rmse = round(np.sqrt(mean_squared_error(target, y_pred_freg)), 2)
    adj_rsquare = round(1-(((1-rsquare)*(len(data)-1))/(len(data)-len(data.columns)-1)), 2)
    fs_model_dict['freg'].append(rsquare)
    fs_model_dict['freg'].append(mean_ae)
    fs_model_dict['freg'].append(median_ae)
    fs_model_dict['freg'].append(mse)
    fs_model_dict['freg'].append(rmse)
    fs_model_dict['freg'].append(adj_rsquare)

    # 3
    y_pred_selk_class = model.fit(data[selk_class_feat], target).predict(data[selk_class_feat])
    rsquare = round(r2_score(target, y_pred_selk_class), 2)
    mean_ae = round(mean_absolute_error(target, y_pred_selk_class), 2)
    median_ae = round(median_absolute_error(target, y_pred_selk_class), 2)
    mse = round(mean_squared_error(target, y_pred_selk_class), 2)
    rmse = round(np.sqrt(mean_squared_error(target, y_pred_selk_class)), 2)
    adj_rsquare = round(1-(((1-rsquare)*(len(data)-1))/(len(data)-len(data.columns)-1)), 2)
    fs_model_dict['selk_class'].append(rsquare)
    fs_model_dict['selk_class'].append(mean_ae)
    fs_model_dict['selk_class'].append(median_ae)
    fs_model_dict['selk_class'].append(mse)
    fs_model_dict['selk_class'].append(rmse)
    fs_model_dict['selk_class'].append(adj_rsquare)

    # 4
    y_pred_selk_reg = model.fit(data[selk_reg_feat], target).predict(data[selk_reg_feat])
    rsquare = round(r2_score(target, y_pred_selk_reg), 2)
    mean_ae = round(mean_absolute_error(target, y_pred_selk_reg), 2)
    median_ae = round(median_absolute_error(target, y_pred_selk_reg), 2)
    mse = round(mean_squared_error(target, y_pred_selk_reg), 2)
    rmse = round(np.sqrt(mean_squared_error(target, y_pred_selk_reg)), 2)
    adj_rsquare = round(1-(((1-rsquare)*(len(data)-1))/(len(data)-len(data.columns)-1)), 2)
    fs_model_dict['selk_reg'].append(rsquare)
    fs_model_dict['selk_reg'].append(mean_ae)
    fs_model_dict['selk_reg'].append(median_ae)
    fs_model_dict['selk_reg'].append(mse)
    fs_model_dict['selk_reg'].append(rmse)
    fs_model_dict['selk_reg'].append(adj_rsquare)

    # 5
    y_pred_sfm_dt = model.fit(data[sfm_dt_feat], target).predict(data[sfm_dt_feat])
    rsquare = round(r2_score(target, y_pred_sfm_dt), 2)
    mean_ae = round(mean_absolute_error(target, y_pred_sfm_dt), 2)
    median_ae = round(median_absolute_error(target, y_pred_sfm_dt), 2)
    mse = round(mean_squared_error(target, y_pred_sfm_dt), 2)
    rmse = round(np.sqrt(mean_squared_error(target, y_pred_sfm_dt)), 2)
    adj_rsquare = round(1-(((1-rsquare)*(len(data)-1))/(len(data)-len(data.columns)-1)), 2)
    fs_model_dict['sfm_dt'].append(rsquare)
    fs_model_dict['sfm_dt'].append(mean_ae)
    fs_model_dict['sfm_dt'].append(median_ae)
    fs_model_dict['sfm_dt'].append(mse)
    fs_model_dict['sfm_dt'].append(rmse)
    fs_model_dict['sfm_dt'].append(adj_rsquare)

    # 6
    y_pred_sfm_rf = model.fit(data[sfm_rf_feat], target).predict(data[sfm_rf_feat])
    rsquare = round(r2_score(target, y_pred_sfm_rf), 2)
    mean_ae = round(mean_absolute_error(target, y_pred_sfm_rf), 2)
    median_ae = round(median_absolute_error(target, y_pred_sfm_rf), 2)
    mse = round(mean_squared_error(target, y_pred_sfm_rf), 2)
    rmse = round(np.sqrt(mean_squared_error(target, y_pred_sfm_rf)), 2)
    adj_rsquare = round(1-(((1-rsquare)*(len(data)-1))/(len(data)-len(data.columns)-1)), 2)
    fs_model_dict['sfm_rf'].append(rsquare)
    fs_model_dict['sfm_rf'].append(mean_ae)
    fs_model_dict['sfm_rf'].append(median_ae)
    fs_model_dict['sfm_rf'].append(mse)
    fs_model_dict['sfm_rf'].append(rmse)
    fs_model_dict['sfm_rf'].append(adj_rsquare)

    # 7
    y_pred_sfm_xgb = model.fit(data[sfm_xgb_feat], target).predict(data[sfm_xgb_feat])
    rsquare = round(r2_score(target, y_pred_sfm_xgb), 2)
    mean_ae = round(mean_absolute_error(target, y_pred_sfm_xgb), 2)
    median_ae = round(median_absolute_error(target, y_pred_sfm_xgb), 2)
    mse = round(mean_squared_error(target, y_pred_sfm_xgb), 2)
    rmse = round(np.sqrt(mean_squared_error(target, y_pred_sfm_xgb)), 2)
    adj_rsquare = round(1-(((1-rsquare)*(len(data)-1))/(len(data)-len(data.columns)-1)), 2)
    fs_model_dict['sfm_xgb'].append(rsquare)
    fs_model_dict['sfm_xgb'].append(mean_ae)
    fs_model_dict['sfm_xgb'].append(median_ae)
    fs_model_dict['sfm_xgb'].append(mse)
    fs_model_dict['sfm_xgb'].append(rmse)
    fs_model_dict['sfm_xgb'].append(adj_rsquare)

    # 8
    y_pred_sfm_log = model.fit(data[sfm_log_feat], target).predict(data[sfm_log_feat])
    rsquare = round(r2_score(target, y_pred_sfm_log), 2)
    mean_ae = round(mean_absolute_error(target, y_pred_sfm_log), 2)
    median_ae = round(median_absolute_error(target, y_pred_sfm_log), 2)
    mse = round(mean_squared_error(target, y_pred_sfm_log), 2)
    rmse = round(np.sqrt(mean_squared_error(target, y_pred_sfm_log)), 2)
    adj_rsquare = round(1-(((1-rsquare)*(len(data)-1))/(len(data)-len(data.columns)-1)), 2)
    fs_model_dict['sfm_log'].append(rsquare)
    fs_model_dict['sfm_log'].append(mean_ae)
    fs_model_dict['sfm_log'].append(median_ae)
    fs_model_dict['sfm_log'].append(mse)
    fs_model_dict['sfm_log'].append(rmse)
    fs_model_dict['sfm_log'].append(adj_rsquare)

    # 9
    y_pred_selp_class = model.fit(data[selp_class_feat], target).predict(data[selp_class_feat])
    rsquare = round(r2_score(target, y_pred_selp_class), 2)
    mean_ae = round(mean_absolute_error(target, y_pred_selp_class), 2)
    median_ae = round(median_absolute_error(target, y_pred_selp_class), 2)
    mse = round(mean_squared_error(target, y_pred_selp_class), 2)
    rmse = round(np.sqrt(mean_squared_error(target, y_pred_selp_class)), 2)
    adj_rsquare = round(1-(((1-rsquare)*(len(data)-1))/(len(data)-len(data.columns)-1)), 2)
    fs_model_dict['selp_class'].append(rsquare)
    fs_model_dict['selp_class'].append(mean_ae)
    fs_model_dict['selp_class'].append(median_ae)
    fs_model_dict['selp_class'].append(mse)
    fs_model_dict['selp_class'].append(rmse)
    fs_model_dict['selp_class'].append(adj_rsquare)

    # 10
    y_pred_selp_reg = model.fit(data[selp_reg_feat], target).predict(data[selp_reg_feat])
    rsquare = round(r2_score(target, y_pred_selp_reg), 2)
    mean_ae = round(mean_absolute_error(target, y_pred_selp_reg), 2)
    median_ae = round(median_absolute_error(target, y_pred_selp_reg), 2)
    mse = round(mean_squared_error(target, y_pred_selp_reg), 2)
    rmse = round(np.sqrt(mean_squared_error(target, y_pred_selp_reg)), 2)
    adj_rsquare = round(1-(((1-rsquare)*(len(data)-1))/(len(data)-len(data.columns)-1)), 2)
    fs_model_dict['selp_reg'].append(rsquare)
    fs_model_dict['selp_reg'].append(mean_ae)
    fs_model_dict['selp_reg'].append(median_ae)
    fs_model_dict['selp_reg'].append(mse)
    fs_model_dict['selp_reg'].append(rmse)
    fs_model_dict['selp_reg'].append(adj_rsquare)

    # 11
    y_pred_vt_50 = model.fit(data[vt_feat_50], target).predict(data[vt_feat_50])
    rsquare = round(r2_score(target, y_pred_vt_50), 2)
    mean_ae = round(mean_absolute_error(target, y_pred_vt_50), 2)
    median_ae = round(median_absolute_error(target, y_pred_vt_50), 2)
    mse = round(mean_squared_error(target, y_pred_vt_50), 2)
    rmse = round(np.sqrt(mean_squared_error(target, y_pred_vt_50)), 2)
    adj_rsquare = round(1-(((1-rsquare)*(len(data)-1))/(len(data)-len(data.columns)-1)), 2)
    fs_model_dict['vt_50'].append(rsquare)
    fs_model_dict['vt_50'].append(mean_ae)
    fs_model_dict['vt_50'].append(median_ae)
    fs_model_dict['vt_50'].append(mse)
    fs_model_dict['vt_50'].append(rmse)
    fs_model_dict['vt_50'].append(adj_rsquare)

    return fs_model_dict