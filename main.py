import streamlit as st
from tensorflow.keras.models import Model , model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_VGG19_model
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_VGG16_model
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inceptionV3_model
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_ResNet50_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_input_ResNet50V2_model
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


prediction_classes = {0:'chicken_noodle',
                     1: 'dumplings',
                     2: 'fried_chicken',
                     3: 'fried_chicken_salad_sticky_rice',
                     4: 'fried_pork_curry_rice',
                     5: 'grilled_pork_with_sticky_rice',
                     6: 'lek_tom_yam',
                     7: 'mama_namtok',
                     8: 'pork_blood_soup',
                     9: 'pork_congee',
                     10: 'pork_suki',
                     11: 'rice_scramble_egg',
                     12: 'rice_topped_with_stir_fried_pork_and_basil',
                     13: 'rice_with_roasted_pork',
                     14: 'roasted_red_pork_noodle',
                     15: 'sliced_grilled_pork_salad',
                     16: 'steamed_rice_with_chicken',
                     17: 'steamed_rice_with_fried_chicken',
                     18: 'stir_fried_rice_noodles_with_chicken',
                     19: 'stir_fried_rice_noodles_with_soy_sauce_and_pork'}

st.title('Food Recommendation Application a Case Study of Thai University')

meal = st.radio("Breakfast or Lunch",('Breakfast', 'Lunch'))
select_weight = st.number_input('Insert weight')
select_height = st.number_input('Insert height')
select_age = st.number_input('Insert age')
select_gender = st.radio("Select gender",('Men', 'Women'))

if meal == 'Lunch':
    st.experimental_singleton.clear()
st.header('Upload/Take an image')
uploaded_file = st.file_uploader("Upload/Take an image", type=["jpg","jpeg"])
if uploaded_file is None:
    st.stop()
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = img.save("img.jpg")
    img_arr = image.load_img("img.jpg", target_size=(224, 224))


st.image(img_arr)
x = image.img_to_array(img_arr)
x = np.expand_dims(x, axis=0)

# load json and create model
select_model = st.radio("Select model",('VGG16_model', 'inceptionV3_model', 'VGG19_model', 'ResNet50V2_model'))
if select_model == 'inceptionV3_model':
    x = preprocess_input_inceptionV3_model(x)
elif select_model == 'VGG16_model':
    x = preprocess_input_VGG16_model(x)
elif select_model == 'VGG19_model':
    x = preprocess_input_VGG19_model(x)
elif select_model == 'ResNet50V2_model':
    x = preprocess_input_ResNet50V2_model(x)

json_file = open(f'models/{select_model}.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(f"models/{select_model}.h5")
print("Loaded model from disk")

y_pred=loaded_model.predict(x,batch_size=1)
y_pred = np.argmax(y_pred)
st.header(f'You are eating: {prediction_classes[y_pred]}')


st.title('Food Recommendation')
# Recommendation part

df = pd.read_csv('Food dataset.csv')
df = df.iloc[5:, :].reset_index(drop = True)
df_nutrient = df.copy()
# Set menus as index
df = df.rename(columns = {'Nutrient': 'index'}).set_index('index')

# Specify Food menu at each index of a list as references when recommending
FOOD_MENUS = df.index.tolist()
CALORIES = df['Calories'].tolist()
CARBOHYDRATES = df['Carbs'].tolist()
PROTEINS = df['Protein'].tolist()
FATS = df['Fat'].tolist()

# Features for recommending
features = df.columns[10:]
# Dataframe with full set of features
features_df = df[features]


# Change types of all columns
features_df = features_df.astype(np.int64)
# Convert dataframe to numpy array for analysis
features_np = features_df.to_numpy()
similarity = cosine_similarity(features_np)


weight = select_weight
size = select_height
age = select_age
gender = select_gender

BMR_MEN =  66.47 + (13.75 * weight) + (5.003 * size) - (6.755 * age)
BMR_WOMAN = 655.1 + (9.563 * weight) + (1.85 * size) - (4.676 * age)
BMR_FORMULA = {'Men': BMR_MEN, 'Women': BMR_WOMAN}


# Requirement / Day in ratio to Calories
CARBOHYDRATE = 0.45
PROTEIN = 0.25
FAT = 0.3
BMR = BMR_FORMULA[gender]

# Calories of each nutrient
CARBOHYDRATE_CALORIES = BMR * CARBOHYDRATE
PROTEIN_CALORIES  = BMR * PROTEIN
FAT_CALORIES = BMR * FAT

# Calories / gram
CARBOHYDRATE_CALORIES_GRAM = 4
PROTEIN_CALORIES_GRAM = 4
FAT_CALORIES_GRAM = 9

# Nutrient Intake (gram/day)
CARBOHYDRATE_INTAKE = CARBOHYDRATE_CALORIES / CARBOHYDRATE_CALORIES_GRAM
PROTEIN_INTAKE = PROTEIN_CALORIES / PROTEIN_CALORIES_GRAM
FAT_INTAKE = FAT_CALORIES / FAT_CALORIES_GRAM

# BMR_WOMAN = 655.1 + (9.563 * weight) + (1.85 * size) - (4.676 * age)

st.write(f'MAX BMR: {round(BMR,2)}')
st.write(f'MAX CARBOHYDRATE_INTAKE: {round(CARBOHYDRATE_INTAKE,2)}')
st.write(f'MAX PROTEIN_INTAKE: {round(PROTEIN_INTAKE,2)}')
st.write(f'MAX FAT_INTAKE: {round(FAT_INTAKE,2)}')

def recommend_foods(food_menu: str, cal_bmr: float, carbs_intake: float, protein_intake: float, fat_intake: float, top: int = 5):
    """
    Recommend food with similarity score :) Best doc-string I could write
    """
    if food_menu not in FOOD_MENUS:
      return 'Food is not in the list.'

    index = np.where(features_df.index == food_menu)[0][0]
    similar_foods = sorted(
              enumerate(similarity[index]),
              key=lambda x:x[1],
              reverse=True
              )[1:]

    # print(similar_foods)

    food_lists = [(FOOD_MENUS[i[0]], i[1], CALORIES[i[0]], CARBOHYDRATES[i[0]], PROTEINS[i[0]], FATS[i[0]]) for i in similar_foods]
    filter_food_lists = [food for food in food_lists if all([float(food[2]) < cal_bmr, float(food[3]) < carbs_intake, float(food[4]) < protein_intake, float(food[5]) < fat_intake])]

    return filter_food_lists[:top+1]


df_nutrient['Carbs'] = df_nutrient['Carbs'].astype(np.float64)
df_nutrient['Protein'] = df_nutrient['Protein'].astype(np.float64)
df_nutrient['Fat'] = df_nutrient['Fat'].astype(np.float64)


if meal == 'Breakfast':
    df_eaten = pd.DataFrame(columns = ['Menu', 'Calories', 'Carbs', 'Protein', 'Fat'])
    df_eaten['Menu'] = [prediction_classes[y_pred]]
    df_eaten['Calories'] = df_nutrient[df_nutrient['Nutrient'] == prediction_classes[y_pred]]['Calories'].values
    df_eaten['Carbs'] = df_nutrient[df_nutrient['Nutrient'] == prediction_classes[y_pred]]['Carbs'].values
    df_eaten['Protein'] = df_nutrient[df_nutrient['Nutrient'] == prediction_classes[y_pred]]['Protein'].values
    df_eaten['Fat'] = df_nutrient[df_nutrient['Nutrient'] == prediction_classes[y_pred]]['Fat'].values
    df_eaten.to_csv('eaten.csv', index = False)
else:
    df_eaten = pd.read_csv('eaten.csv')
    df_eaten = df_eaten.append({'Menu':prediction_classes[y_pred],'Calories': df_nutrient[df_nutrient['Nutrient'] == prediction_classes[y_pred]]['Calories'].values[0], 'Carbs': df_nutrient[df_nutrient['Nutrient'] == prediction_classes[y_pred]]['Carbs'].values[0], 'Protein':df_nutrient[df_nutrient['Nutrient'] == prediction_classes[y_pred]]['Protein'].values[0],'Fat':df_nutrient[df_nutrient['Nutrient'] == prediction_classes[y_pred]]['Fat'].values[0]}, ignore_index=True)


st.write(df_eaten)
last_Calories = df_eaten['Calories'].sum()
last_Carbs = df_eaten['Carbs'].sum()
last_Protein = df_eaten['Protein'].sum()
last_Fat = df_eaten['Fat'].sum()


recommended = recommend_foods(prediction_classes[y_pred], BMR-last_Calories, CARBOHYDRATE_INTAKE-last_Carbs, PROTEIN_INTAKE-last_Protein, FAT_INTAKE-last_Fat, 5)
if len(recommended) > 0:
    st.header(f'Top {len(recommended)} Recommended Foods')


    for i in range(len(recommended)):
        img_arr = image.load_img(f"food_images/{recommended[i][0]}.jpeg", target_size=(224, 224))
        col1, col2 = st.columns(2)
        with col1:
            st.write(recommended[i][0])
            st.image(img_arr)
        with col2:
            st.write(f'Recommendation percentage: {round(float(recommended[i][1]),2)}')
            st.write(f'Calories: {round(float(recommended[i][2]), 2)}')
            st.write(f'Carbs (grams): {round(float(recommended[i][3]), 2)}')
            st.write(f'Protein (grams): {round(float(recommended[i][4]), 2)}')
            st.write(f'Fat: (grams) {round(float(recommended[i][5]), 2)}')
elif len(recommended) == 0:
    st.header('You have exceed your nutrient goals for today. Please eat less :)')


