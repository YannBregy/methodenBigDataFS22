
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import pickle

# Defining some general properties of the app
st.set_page_config(
    page_title= "AirBNB price predictor",
    page_icon = "🛏️",
    layout="wide"
    )

# Define Load functions
@st.cache()
def load_data():
    data = pd.read_csv("listings_complete.csv")
    return data

@st.cache(allow_output_mutation=True)
def load_model():
    filename = "final_model.joblib"
    loaded_model = joblib.load(filename)
    return(loaded_model)

# Load Data and Model
data = load_data()
model = load_model()


### Header of app ###

st.title("New York's AirBNB price predictor")
st.markdown("This application is an AirBNB dashboard where you can find out more about New York's listings. "\
            "Furthermore, you can upload your own listing to see what it is worth according to our predictions!")

#### Section for listing overview ###

st.header("Overview of listings")
st.markdown("In this section, you get an overview of existing listings in New York. You can select your desired"\
            " filters to restrict the data to your liking. ")

## Introducing three colums for user inputs ##
row1_col1, row1_col2, row1_col3 = st.columns([1,1,1])

# Price slider, default is at min and max so that all listings are included and can then be manually reduced
price = row1_col1.slider("Price of the listing",
                  data["price"].min(),
                  data["price"].max(),
                  (float(data["price"].min()), float(data["price"].max())), step=5.0)

# Selectbox to choose the apartment type. By default, all apartment types are shown
types = ["All"]
types.extend(data["room_type"].unique())
type = row1_col2.selectbox("Select apartment type", types)

# Selectbox to choose neighbourhood group. By default, all neighbourhood groups are selected
neighbourhood_groups = ["All"]
neighbourhood_groups.extend(data["neighbourhood_group_cleansed"].unique())
neighbourhood_group = row1_col3.selectbox("Select neighbourhood", neighbourhood_groups)
# If a neighbourhood is chosen, a more detailed neighbourhood can be selected
if neighbourhood_group != "All":
    neighbourhoods = ["All"]
    filtered_neigbhourdoods = data.loc[data["neighbourhood_group_cleansed"] == neighbourhood_group, :]
    neighbourhoods.extend(filtered_neigbhourdoods["neighbourhood_cleansed"].unique())
    neighbourhood = row1_col3.selectbox("Select detailed neighbourhood", neighbourhoods)

# Creating filtered data set according to inputs from first columns
if neighbourhood_group == "All":
    filtered_data = data.loc[(data["price"] >= price[0]) &
                         (data["price"] <= price[1]) &
                         (data["room_type"] == type if type != "All" else True),:]
                        # Room type is filtered with one-line if statement for reading purposes
else:
    if neighbourhood == "All":
        filtered_data = data.loc[(data["price"] >= price[0]) &
                             (data["price"] <= price[1]) &
                             (data["room_type"] == type if type != "All" else True) &
                             (data["neighbourhood_group_cleansed"] == neighbourhood_group), :]
    else:
        filtered_data = data.loc[(data["price"] >= price[0]) &
                                 (data["price"] <= price[1]) &
                                 (data["room_type"] == type if type != "All" else True) &
                                 (data["neighbourhood_group_cleansed"] == neighbourhood_group) &
                                 (data["neighbourhood_cleansed"] == neighbourhood), :]



## Creating two columns on second row to display filtered data ##
row2_col1, row2_col2  = st.columns([1,1])

# Display map of listings
row2_col1.map(filtered_data)

# Display various information about filtered data in column2
row2_col2.markdown("Your filtered data contains {0} listings out of {1} total listings.".format(len(filtered_data),
                                                                                                len(data)))
# When checked, display detailed dataset of filtered data
if row2_col2.checkbox("Show filtered results in detail", False):
    row2_col2.subheader("Raw Data")
    row2_col2.write(filtered_data)

# When checked, display graph of price prediction accuracy for filtered data
if row2_col2.checkbox("Show prediction graph about filtered results", False):
    highest = max(filtered_data[["price", "predicted_price"]].max())

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.scatter(filtered_data["price"], filtered_data["predicted_price"], color='blue')

    ax1.plot([0, highest], [0, highest],
             lw=3, color="green", label="Trendline")
    ax1.set_title("Actual vs Predicted Price Plot")
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.legend()
    row2_col2.pyplot(fig1, use_container_width=True)

# When checked, display average values of filtered data and compare them with the whole dataset
if row2_col2.checkbox("Show average values for filtered results", True):
    with row2_col2.container():
        st.metric(label="Average price", value="{0}".format(round(filtered_data["price"].mean())),
                         delta="{0}$ from mean".format(round(filtered_data["price"].mean()-data["price"].mean())))
        st.metric(label="Average minimum nights", value="{0}".format(round(filtered_data["minimum_nights"].mean())),
                         delta="{0} from mean".format(
                             round(filtered_data["minimum_nights"].mean() - data["minimum_nights"].mean(), 1)))
        st.metric(label="Average review rating", value="{0}"
                  .format(round(filtered_data["review_scores_rating"].mean(), 2)), delta="{0} from mean"
                  .format(round(filtered_data["review_scores_rating"].mean() - data["review_scores_rating"].mean(), 2)))


## Differentiate between landlord and renter for further analysis ##
row3_col1, row3_col2 = st.columns([1,1])
landlord = row3_col1.checkbox("I'm a landlord who wants to know what my property would price on AirBNB!")
renter = row3_col2.checkbox("Im a renter looking to see if the property that I'm considering is overvalued!")

# The landlord analysis is for owners of a property that want to know what their listing is worth and how to improve
# its price
if landlord:
    ### Section for statistics on easily improvable features of the listing ###
    st.header("Find ways to easily improve your price!")
    st.markdown("There a multiple simple tricks that allow you to improve your price wihtout too much effort.")

    # Divide section into three columns to display graphs
    row4_col1, row4_col2, row4_col3 = st.columns([1, 1, 1])
    # Create barchart with filtered data about superhost
    barplotdata = filtered_data.groupby("host_is_superhost", as_index=False)["price"].mean()
    barplotdata = barplotdata.rename(index={1: 'Superhost', 0: 'Not a superhost'})
    fig2, ax2 = plt.subplots(figsize=(8,3.7))
    ax2.bar(barplotdata.index.astype(str), barplotdata["price"], color = "#fc8d62")
    ax2.set_ylabel("Price")
    # Display chart
    row4_col1.pyplot(fig2)

    # Create barchart with filtered data about verification of host
    barplotdata = filtered_data.groupby("host_identity_verified", as_index=False)["price"].mean()
    barplotdata = barplotdata.rename(index={1: 'Identity verified', 0: 'Identity not verified'})
    fig3, ax3 = plt.subplots(figsize=(8, 3.7))
    ax3.bar(barplotdata.index.astype(str), barplotdata["price"], color="green")
    ax3.set_ylabel("Price")
    # Display chart
    row4_col2.pyplot(fig3)

    # Create barchart with filtered data about host profile picture
    barplotdata = filtered_data.groupby("host_has_profile_pic", as_index=False)["price"].mean()
    barplotdata = barplotdata.rename(index={1: 'Host has profile picture', 0: 'Host does not have profile pricutre'})
    fig4, ax4 = plt.subplots(figsize=(8, 3.7))
    ax4.bar(barplotdata.index.astype(str), barplotdata["price"], color="blue")
    ax4.set_ylabel("Price")
    # Display chart
    row4_col3.pyplot(fig4)

    st.markdown("You can see that simple steps such as getting verified, adding a profile picture to your account"\
                " or becoming superhost can marginally improve your listing's price!")

    ### Section for listing price prediction ###
    st.header("Predicting your listing's price")
    uploaded_data = st.file_uploader("Choose a file with your listing data to predict price")
    st.markdown("If you want to test, we have prepared a couple example listings to test out. Simply click to download!")
    row6_col1, row6_col2, row6_col3, row6_col4 = st.columns([1,1,1,1])
    example1 = pd.read_csv("test_example1.csv").to_csv()
    row6_col1.download_button(
        label="Download example 1",
        data=example1,
        file_name='example1.csv',
        mime='text/csv')
    example2 = pd.read_csv("test_example2.csv").to_csv()
    row6_col2.download_button(
        label="Download example 2",
        data=example2,
        file_name='example2.csv',
        mime='text/csv')
    example3 = pd.read_csv("test_example3.csv").to_csv()
    row6_col3.download_button(
        label="Download example 3",
        data=example3,
        file_name='example3.csv',
        mime='text/csv')
    example4 = pd.read_csv("test_example4.csv").to_csv()
    row6_col4.download_button(
        label="Download example 4",
        data=example4,
        file_name='example4.csv',
        mime='text/csv')

    # Add action to be done if file is uploaded
    if uploaded_data is not None:
        # Getting Data and Making Predictions
        new_listing = pd.read_csv(uploaded_data)
        new_listing_for_model = new_listing.drop(["listing_url", "host_name", "latitude",
                                                  "longitude", "neighbourhood_cleansed",
                                                  "neighbourhood_group_cleansed", "room_type",
                                                  "picture_url"],axis=1)
        new_listing["predicted_price"] = model.predict(new_listing_for_model)
        first_listing = new_listing.head(1)

        predicted_price = first_listing.loc[0, "predicted_price"]
        st.success("You listing is worth ${0}!".format(predicted_price))

        is_superhost = first_listing.loc[0, "host_is_superhost"]
        is_verified = first_listing.loc[0, "host_identity_verified"]
        has_profile_pic = first_listing.loc[0, "host_has_profile_pic"]
        number_accommodates = first_listing.loc[0, "accommodates"]

        row5_col1, row5_col2, row5_col3, row5_col4 = st.columns([1,1,1,1])

        row5_col1.image("https://ih1.redbubble.net/image.2704361841.4497/st,small,845x845-pad,1000x1000,f8f8f8.jpg")
        row5_col2.image("https://miro.medium.com/max/800/1*fFUnF8o4URvCowXSacCgGA.jpeg")
        row5_col3.image("https://t3.ftcdn.net/jpg/03/46/83/96/360_F_346839683_6nAPzbhpSkIpb8pmAwufkC7c5eD7wYws.jpg")
        row5_col4.image("https://st3.depositphotos.com/1432405/13041/v/950/depositphotos_130410922-stock-illustration"\
                        "-bed-icon-cartoon-style.jpg")

        if is_superhost == 0:
            superhost_first_listing = first_listing
            superhost_first_listing["host_is_superhost"] = 1
            superhost_first_listing["predicted_price"] = model.predict(superhost_first_listing.drop(
                ["predicted_price", "listing_url", "host_name", "latitude", "longitude",
                 "neighbourhood_cleansed","neighbourhood_group_cleansed", "room_type",
                 "picture_url"], axis=1))
            superhost_predicted_price = superhost_first_listing.loc[0, "predicted_price"]
            if (superhost_predicted_price - predicted_price) > 0:
                row5_col1.markdown("By becoming superhost, you could improve your price by {0}\$!"
                        .format(round(superhost_predicted_price - predicted_price), 1))
            else:
                row5_col1.markdown("It seems that in your case, becoming a superhost would actually lower your price.")
        else:
            row5_col1.markdown("Good job, you are already a superhost!")

        if is_verified == 0:
            verified_first_listing = first_listing
            verified_first_listing["host_identity_verified"] = 1
            verified_first_listing["predicted_price"] = model.predict(verified_first_listing.drop(
                ["predicted_price", "listing_url", "host_name", "latitude", "longitude",
                 "neighbourhood_cleansed", "neighbourhood_group_cleansed", "room_type",
                 "picture_url"], axis=1))
            verified_predicted_price = verified_first_listing.loc[0, "predicted_price"]
            if (verified_predicted_price - predicted_price) > 0:
                row5_col2.markdown("By verifying your account, you could improve your price by {0}\$!"
                        .format(round(verified_predicted_price - predicted_price), 1))
            else:
                row5_col2.markdown("It seems that in your case, verifying your account would actually lower your price.")
        else:
            row5_col2.markdown("Good job, you have already verified your acount!")

        if has_profile_pic == 0:
            pic_first_listing = first_listing
            pic_first_listing["host_identity_verified"] = 1
            pic_first_listing["predicted_price"] = model.predict(pic_first_listing.drop(
                ["predicted_price", "listing_url", "host_name", "latitude", "longitude",
                 "neighbourhood_cleansed", "neighbourhood_group_cleansed", "room_type",
                 "picture_url"], axis=1))
            pic_predicted_price = pic_first_listing.loc[0, "predicted_price"]
            if (pic_predicted_price - predicted_price) > 0:
                row5_col3.markdown("By adding a profile picture to your account, you could improve your price by {0}\$!"
                        .format(round(pic_predicted_price - predicted_price), 1))
            else:
                row5_col3.markdown(
                    "It seems that in your case, adding a profile picture would actually lower your price.")
        else:
            row5_col3.markdown("Good job, you have already added a profile picture to your account!")

        accommodates_first_listing = first_listing
        accommodates_first_listing["accommodates"] = number_accommodates + 1
        accommodates_first_listing["predicted_price"] = model.predict(accommodates_first_listing.drop(
            ["predicted_price", "listing_url", "host_name", "latitude", "longitude",
             "neighbourhood_cleansed", "neighbourhood_group_cleansed", "room_type",
             "picture_url"], axis=1))
        accommodates_predicted_price = accommodates_first_listing.loc[0, "predicted_price"]
        if (accommodates_predicted_price - predicted_price) > 0:
            row5_col4.markdown("By adding a spare bed or a sofa couch to accommodate one more person,"\
                        " you could improve your price by {0}\$!"
                        .format(round(accommodates_predicted_price - predicted_price), 1))
        else:
            row5_col4.markdown("It seems that in your case, adding a bed would not improve your price.")



# The renters analysis allows renters to check if a searched for listing is valued accurately
if renter:
    ### Section for checking some exact listing to see if it is overvalued ###
    st.header("Check if the listing you are considering is valued correctly")
    input_link = st.text_input("Enter the URL of the listing you want to check",
                               value="Enter URL here, for example https://www.airbnb.com/rooms/9357")
    st.markdown("A couple examples if you need inspiration: 11420840 / 22747209 / 7756711 / 24449621")
    if input_link != "Enter URL here":
        try:
            listing_to_check = data.loc[data["listing_url"] == input_link]
            check_predicted_price = listing_to_check.iloc[0]['predicted_price']
            check_price = listing_to_check.iloc[0]['price']
            check_host = listing_to_check.iloc[0]['host_name']
            check_picture = listing_to_check.iloc[0]['picture_url']
        except IndexError:
            st.markdown("Your listing could not be found in our database, try changing the URL.")
        else:
            row5_col1, row5_col2 = st.columns([1, 1])
            with row5_col1.container():
                st.image(check_picture, width=600)
            row5_col2.subheader("The AirBNB is hosted by {0}".format(check_host))
            if check_price > check_predicted_price:
                row5_col2.markdown("Your listing is valued at {0}\$, but our predictions value it at {1}\$."
                                   .format(round(check_price), round(check_predicted_price)))
                row5_col2.subheader("You might be overpaying, so watch out!")
            else:
                row5_col2.markdown("Your listing is valued at {0}\$ and our predictions value it at {1}\$."
                                   .format(round(check_price), round(check_predicted_price)))
                row5_col2.subheader("It seems like you found yourself a catch!")


























