import uvicorn
import joblib
import xgboost
import pandas as pd
import numpy as np
import json
import pickle
from flask import jsonify
from fastapi import FastAPI, File, UploadFile
from sklearn.base import BaseEstimator, TransformerMixin
from fastapi.responses import JSONResponse


app = FastAPI(
              title="AccountCode Model API",
              description="A simple API that use XGBoost model to predict the account code of invoice given the features as input in csv"
)

'''Load xgboost model pickle'''
xgb_modelfile = open('/code/app/model/XGBoost_tuned.pkl',"rb")
xgb_model = pickle.load(xgb_modelfile)

'''Load labels dict'''
label_dict= open("/code/app/model/label_mapping.json")
true_label_dict = json.load(label_dict)

'''Load feature dict files'''
pickle_product = open("/code/app/model/product.pickle","rb")
product = pickle.load(pickle_product)

pickle_amount = open("/code/app/model/amount.pickle","rb")
amount = pickle.load(pickle_amount)

pickle_price = open("/code/app/model/price.pickle","rb")
price = pickle.load(pickle_price)

pickle_unit = open("/code/app/model/unit.pickle","rb")
unit = pickle.load(pickle_unit)

pickle_tax = open("/code/app/model/tax.pickle","rb")
tax = pickle.load(pickle_tax)

pickle_invoiceid = open("/code/app/model/invoiceid.pickle","rb")
invoiceid = pickle.load(pickle_invoiceid)

pickle_bodyid = open("/code/app/model/bodyid.pickle","rb")
bodyid = pickle.load(pickle_bodyid)

pickle_invoicestatusid = open("/code/app/model/invoicestatusid.pickle","rb")
invoicestatusid = pickle.load(pickle_invoicestatusid)

pickle_customername = open("/code/app/model/customername.pickle","rb")
customername = pickle.load(pickle_customername)

pickle_currencycode = open("/code/app/model/currencycode.pickle","rb")
currencycode = pickle.load(pickle_currencycode)

pickle_vat_deducation = open("/code/app/model/vat_deducation.pickle","rb")
vat_deducation = pickle.load(pickle_vat_deducation)

pickle_vat_status = open("/code/app/model/vat_status.pickle","rb")
vat_status = pickle.load(pickle_vat_status)



##read to dict
product_dict = dict((v,k) for k,v in product.items())
amount_dict = dict((v,k) for k,v in amount.items())
price_dict = dict((v,k) for k,v in price.items())
unit_dict = dict((v,k) for k,v in unit.items())
tax_dict = dict((v,k) for k,v in tax.items())
invoiceid_dict = dict((v,k) for k,v in invoiceid.items())
bodyid_dict = dict((v,k) for k,v in bodyid.items())
invoicestatusid_dict = dict((v,k) for k,v in invoicestatusid.items())
customername_dict = dict((v,k) for k,v in customername.items())
currencycode_dict = dict((v,k) for k,v in currencycode.items())
vat_deducation_dict = dict((v,k) for k,v in vat_deducation.items())
vat_status_dict = dict((v,k) for k,v in vat_status.items())


def strip(x):
  return  str(x).split('.')[0]

def price(x):
  return  str(x*100)

@app.post("/predict-account-file")
async def predict_code(csv_file: UploadFile = File(...)):
    """
    A simple function that receives invoice data and predicts the account code of the invoice.
    input: invoice csv and
    return: account code
    """
    userData = pd.read_csv(csv_file.file)
    userData = userData.drop(['id'], axis=1)

    userData['amount'] = userData['amount'].apply(lambda x: strip(x))
    userData['tax'] = userData['tax'].apply(lambda x: strip(x))
    userData['vat_deducation'] = userData['vat_deducation'].apply(lambda x: strip(x))
    userData['price'] = userData['price'].apply(lambda x: price(x))

    userData['billdate'] = pd.to_datetime(userData['billdate'])
    userData['billdate_year'] = userData['billdate'].dt.year
    userData['billdate_month'] = userData['billdate'].dt.month
    userData['billdate_day'] = userData['billdate'].dt.day
    userData = userData.drop(['billdate'], axis=1)

    # handle values that do not exist in respective dictionaries
    userData['unit'] = userData["unit"].fillna("null")
    userData['product'] = userData['product'].replace('-', "noproduct", regex=True)

    ##remove special characters in customername and product columns
    spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
                  "*", "+", ",", "-", ".", "/", ":", ";", "<",
                  "=", ">", "?", "@", "[", "\\", "]", "^", "_",
                  "`", "{", "|", "}", "~", "–"]
    for char in spec_chars:
        userData['product'] = userData['product'].str.replace(char, ' ')
        userData['customername'] = userData['customername'].str.replace(char, ' ')

    ##remove white space and convert to lowecase
    userData['product'] = userData['product'].str.replace(' ', '').str.lower()
    userData['customername'] = userData['customername'].str.replace(' ', '').str.lower()

    #map categorical variables to dicts loaded from pickle files
    userData = userData.replace({"product": product_dict, "amount": amount_dict, "price": price_dict,
                                 "unit":unit_dict, "tax": tax_dict, "invoiceid": invoiceid_dict, "bodyid": bodyid_dict,
                                 "invoicestatusid": invoicestatusid_dict,"customername": customername_dict,
                                 "currencycode": currencycode_dict, "vat_deducation": vat_deducation_dict,
                                 "vat_status": vat_status_dict})
    ###predict on new user data
    account_code = xgb_model.predict(userData)
    account_code = [k for k, v in true_label_dict.items() if v in account_code]
    ac_dump = JSONResponse({'account_code': account_code})
    return ac_dump


if __name__ == '__main__':
    uvicorn.run(app,host="0.0.0.0",port=8000)
