import streamlit as st
import numpy as np
import pickle
import random

st.set_page_config(layout="wide", page_title="Loan Prediction App")

# ==========================
# LOAD MODEL
# ==========================
model = pickle.load(open("model.pkl", "rb"))

# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("📘 Informasi Aplikasi")
st.sidebar.write("""
Aplikasi ini memprediksi apakah suatu pinjaman **Good Loan** atau **Bad Loan**
berdasarkan 59 fitur yang telah diproses sesuai model training.
""")

st.sidebar.write("⚡ Dibuat dengan Streamlit + TensorFlow")

# ==========================
# TITLE
# ==========================
st.title("💼 Loan Default Prediction")
st.caption("Masukkan nilai fitur atau gunakan tombol _Generate Random_ untuk mengisi otomatis.")

# ==========================
# FIXED ENCODING MAPPINGS
# ==========================

map_term = {"36 months": 0, "60 months": 1}
map_term_list = list(map_term.keys())

map_grade = {"A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"G":6}
grade_list = list(map_grade.keys())

sub_grade_categories = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
map_sub_grade = {cat: i for i, cat in enumerate(sub_grade_categories)}

credit_history_categories = ["+5 years","1-3 years","3-5 years","<1 year"]
map_credit_history = {cat: i for i, cat in enumerate(credit_history_categories)}

purpose_categories = [
    "credit_card","debt_consolidation","educational","home_improvement",
    "house","major_purchase","medical","moving","other","renewable_energy",
    "small_business","vacation","wedding"
]

verification_status_categories = ["Source Verified","Verified"]

emp_length_categories = [
    "10+ years","2 years","3 years","4 years","5 years",
    "6 years","7 years","8 years","9 years","< 1 year"
]

home_ownership_categories = ["MORTGAGE","NONE","OTHER","OWN","RENT"]

def encode_one_hot(value, categories):
    arr = np.zeros(len(categories))
    if value in categories:
        arr[categories.index(value)] = 1
    return arr

# ==========================
# RANDOM INPUT GENERATION
# ==========================

def generate_random():
    return {
        "term": random.choice(map_term_list),
        "grade": random.choice(grade_list),
        "sub_grade": random.choice(sub_grade_categories),
        "initial_list_status": random.choice(["f", "w"]),
        "credit_history": random.choice(credit_history_categories),
        "purpose": random.choice(purpose_categories),
        "verification_status": random.choice(["Source Verified", "Verified"]),
        "emp_length": random.choice([
            "< 1 year","1 year","2 year","3 year","4 year","5 year",
            "6 year","7 year","8 year","9 year","10+ year"
        ]),
        "home_ownership": random.choice(["RENT","OWN","MORTAGE","OTHER","NONE"]),
        "loan_amnt": random.randint(500, 50000),
        "int_rate": round(random.uniform(5, 25), 2),
        "annual_inc": random.randint(10000, 200000),
        "dti": round(random.uniform(0, 40), 2),
        "revol_bal": random.randint(0, 50000),
        "revol_util": round(random.uniform(0, 100), 2),
        "delinq_2yrs": random.randint(0, 5),
        "inq_last_6mths": random.randint(0, 10),
        "open_acc": random.randint(1, 20),
        "pub_rec": random.randint(0, 5),
        "total_acc": random.randint(5, 40),
        "out_prncp": random.randint(0, 50000),
        "total_pymnt": random.randint(0, 50000),
        "total_rec_int": random.randint(0, 20000),
        "total_rec_late_fee": random.randint(0, 500),
        "recoveries": random.randint(0, 5000),
        "collection_recovery_fee": random.randint(0, 1000),
        "last_pymnt_amnt": random.randint(0, 20000),
    }

# ==========================
# FORM INPUT
# ==========================

st.subheader("📝 Input Fitur")

random_button = st.button("🎲 Generate Random Input")

if random_button:
    rand = generate_random()
else:
    rand = None

col1, col2, col3 = st.columns(3)

with col1:
    term = st.selectbox("Term", map_term_list, index=map_term_list.index(rand["term"]) if rand else 0)
    grade = st.selectbox("Grade", grade_list, index=grade_list.index(rand["grade"]) if rand else 0)
    sub_grade = st.selectbox("Sub Grade", sub_grade_categories, index=sub_grade_categories.index(rand["sub_grade"]) if rand else 0)
    initial_list_status = st.selectbox("Initial List Status", ["f", "w"], index=["f", "w"].index(rand["initial_list_status"]) if rand else 0)
    credit_history = st.selectbox("Credit History", credit_history_categories, index=credit_history_categories.index(rand["credit_history"]) if rand else 0)
    purpose = st.selectbox("Purpose", purpose_categories, index=purpose_categories.index(rand["purpose"]) if rand else 0)

with col2:
    verification_status = st.selectbox("Verification Status", 
                                       ["Source Verified", "Verified"], 
                                       index=["Source Verified","Verified"].index(rand["verification_status"]) if rand else 0)

    emp_length = st.selectbox("Employment Length", 
                              ["< 1 year","1 year","2 year","3 year","4 year","5 year",
                               "6 year","7 year","8 year","9 year","10+ year"],
                              index=["< 1 year","1 year","2 year","3 year","4 year","5 year",
                                     "6 year","7 year","8 year","9 year","10+ year"].index(rand["emp_length"]) if rand else 0)

    home_ownership = st.selectbox("Home Ownership", 
                                  ["RENT","OWN","MORTAGE","OTHER","NONE"],
                                  index=["RENT","OWN","MORTAGE","OTHER","NONE"].index(rand["home_ownership"]) if rand else 0)

    loan_amnt = st.number_input("Loan Amount", min_value=0.0, value=float(rand["loan_amnt"]) if rand else 0.0)
    int_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=float(rand["int_rate"]) if rand else 0.0)
    annual_inc = st.number_input("Annual Income", min_value=0.0, value=float(rand["annual_inc"]) if rand else 0.0)
    dti = st.number_input("DTI", min_value=0.0, value=float(rand["dti"]) if rand else 0.0)
    revol_bal = st.number_input("Revolving Balance", min_value=0.0, value=float(rand["revol_bal"]) if rand else 0.0)
    revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, value=float(rand["revol_util"]) if rand else 0.0)

with col3:
    delinq_2yrs = st.number_input("Delinquency (2 yrs)", min_value=0.0, value=float(rand["delinq_2yrs"]) if rand else 0.0)
    inq_last_6mths = st.number_input("Inquiries (6 months)", min_value=0.0, value=float(rand["inq_last_6mths"]) if rand else 0.0)
    open_acc = st.number_input("Open Accounts", min_value=0.0, value=float(rand["open_acc"]) if rand else 0.0)
    pub_rec = st.number_input("Public Records", min_value=0.0, value=float(rand["pub_rec"]) if rand else 0.0)
    total_acc = st.number_input("Total Accounts", min_value=0.0, value=float(rand["total_acc"]) if rand else 0.0)

    out_prncp = st.number_input("Outstanding Principal", min_value=0.0, value=float(rand["out_prncp"]) if rand else 0.0)
    total_pymnt = st.number_input("Total Payment", min_value=0.0, value=float(rand["total_pymnt"]) if rand else 0.0)
    total_rec_int = st.number_input("Total Received Interest", min_value=0.0, value=float(rand["total_rec_int"]) if rand else 0.0)
    total_rec_late_fee = st.number_input("Total Late Fee", min_value=0.0, value=float(rand["total_rec_late_fee"]) if rand else 0.0)
    recoveries = st.number_input("Recoveries", min_value=0.0, value=float(rand["recoveries"]) if rand else 0.0)
    collection_recovery_fee = st.number_input("Collection Recovery Fee", min_value=0.0, value=float(rand["collection_recovery_fee"]) if rand else 0.0)
    last_pymnt_amnt = st.number_input("Last Payment Amount", min_value=0.0, value=float(rand["last_pymnt_amnt"]) if rand else 0.0)

# Additional numeric defaults
collections_12_mths_ex_med = 0
acc_now_delinq = 0
tot_coll_amt = 0
tot_cur_bal = 0
total_rev_hi_lim = 0
id_default = 0

# ==========================
# BUILD FEATURE ARRAY
# ==========================

if st.button("🚀 Prediksi"):
    
    term_enc = map_term[term]
    grade_enc = map_grade[grade]
    sub_grade_enc = map_sub_grade[sub_grade]
    initial_list_status_enc = 0 if initial_list_status == "f" else 1
    credit_history_enc = map_credit_history[credit_history]

    emp_map = {
        "< 1 year": "< 1 year",
        "1 year": "2 years",
        "2 year": "2 years",
        "3 year": "3 years",
        "4 year": "4 years",
        "5 year": "5 years",
        "6 year": "6 years",
        "7 year": "7 years",
        "8 year": "8 years",
        "9 year": "9 years",
        "10+ year": "10+ years"
    }

    emp_ohe = encode_one_hot(emp_map[emp_length], emp_length_categories)
    purpose_ohe = encode_one_hot(purpose, purpose_categories)
    verification_ohe = encode_one_hot(verification_status, verification_status_categories)
    home_ohe = encode_one_hot(home_ownership.replace("MORTAGE","MORTGAGE"), home_ownership_categories)

    features = np.concatenate([
        [
            term_enc, grade_enc, sub_grade_enc, initial_list_status_enc,
            id_default,
            loan_amnt, int_rate, annual_inc, dti, delinq_2yrs,
            inq_last_6mths, open_acc, pub_rec, revol_bal, revol_util,
            total_acc, out_prncp, total_pymnt, total_rec_int,
            total_rec_late_fee, recoveries, collection_recovery_fee,
            last_pymnt_amnt, collections_12_mths_ex_med, acc_now_delinq,
            tot_coll_amt, tot_cur_bal, total_rev_hi_lim,
            credit_history_enc
        ],
        purpose_ohe,
        verification_ohe,
        emp_ohe,
        home_ohe
    ])

    x = features.reshape(1, -1)

    st.info(f"Total fitur: **{x.shape[1]} / 59**")

    # ======== AUTO-DETECT PREDICTION ========
    raw_pred = model.predict(x)

    if raw_pred.shape == (1, 1):
        prob = raw_pred[0][0]
    elif raw_pred.shape == (1,):
        prob = raw_pred[0]
    elif raw_pred.shape == (1, 2):
        prob = raw_pred[0][1]
    else:
        prob = float(raw_pred.flatten()[0])

    pred = 1 if prob >= 0.5 else 0

    st.subheader("📊 Hasil Prediksi")

    # -------------------------------
    # Result UI Card
    # -------------------------------
    if pred == 1:
        st.success(f"### ✔ GOOD LOAN \n Probability of Good Loan: **{prob:.3f}**")
    else:
        st.error(f"### ❌ BAD LOAN \n Probability of Bad Loan: **{1-prob:.3f}**")