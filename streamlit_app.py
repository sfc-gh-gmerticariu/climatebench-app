'''
Run a ClimateBench emulator in your browser!
'''
# This was derived from this example: https://github.com/streamlit/demo-self-driving

import streamlit as st
import numpy as np
import tensorflow as tf

import xarray as xr
import geoviews as gv
import geoviews.feature as gf
from cartopy import crs as ccrs

max_co2 = 9500.
max_ch4 = 0.8
max_so2 = 90.
max_bc = 9.


def normalize_inputs(data):
    return np.asarray(data) / np.asarray([max_co2, max_ch4, max_so2, max_so2])


def unnormalize_outputs(data):
    return np.asarray(data) * np.asarray([max_co2, max_ch4, max_so2, max_so2])


def global_mean(ds):
    weights = np.cos(np.deg2rad(ds.latitude))
    return ds.weighted(weights).mean(['latitude', 'longitude'])


def download_file(file_path):
    import os
    import urllib
    import tarfile

    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

        if file_path.endswith('.tar.gz'):
            tarfile.open(file_path).extractall()

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


# This is the main app app itself, which appears when the user selects "Run the app".
def main():

    # Draw the UI element to select parameters for ClimateBench.
    co2, ch4, so2, bc = emissions_ui()

    # Download external dependencies.
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    # Get the temperature  and uncertainty from the emulator
    temperature, uncertainty = climatebench_gp(co2, ch4, so2, bc)
    x, y = np.linspace(0, 360, 144), np.linspace(-90, 90, 96)
    dataset = xr.DataArray(temperature, coords={'latitude': (('latitude',), y), 'longitude': (('longitude',), x)})

    # Draw the header and image.
    st.subheader("Real-time Climate Simulations")
    # st.markdown("**ClimateBench Emulator** (CO2 `%3.1f`) (CH4 `%3.1f`)" % (co2, ch4))
    st.markdown(f"Global mean temperature change: {global_mean(dataset):3.1f}K +/- {uncertainty:3.1f}K")


    air_temperature = gv.Dataset(dataset, ['longitude', 'latitude'], 'air_temperature')
    fig = gv.render((air_temperature.to.image().opts(tools=['hover'], cmap="coolwarm", clim=(-6., 6.)) * \
                    gf.coastline().opts(line_color='black', width=600, height=380)).opts(projection=ccrs.Robinson()))

    st.bokeh_chart(fig, use_container_width=True)

# This sidebar UI lets the user select parameters for ClimateBench.
def emissions_ui():
    st.sidebar.markdown("# Emissions")
    co2 = st.sidebar.slider("CO2 concentrations (GtCO2)", 0.0, max_co2, 1800., 10.)
    ch4 = st.sidebar.slider("Methane emissions (GtCH4 / year)", 0.0, max_ch4, 0.3, 0.005)
    #  Just use global mean values for aerosol for simplicity
    so2 = st.sidebar.slider("SO2 emissions (TgSO2 / year)", 0.0, max_so2, 85., 1.)
    bc = st.sidebar.slider("BC emissions (TgBC / year)", 0.0, max_bc, 7., 0.1)
    return normalize_inputs([co2, ch4, so2, bc])


# Run the GP model
def climatebench_gp(co2, ch4, so2, bc):
    # Load the model. Because this is cached it will only happen once.
    @st.cache(allow_output_mutation=True)
    def load_model(model_path):
        loaded_model = tf.saved_model.load(model_path)
        return loaded_model
    model = load_model("climatebench_webapp")

    # Run the ClimateBench emulator
    inputs = tf.convert_to_tensor([[co2, ch4, so2, bc]], dtype=tf.float64)
    posterior_mean, posterior_variance = model.predict_f_compiled(inputs)

    posterior_tas = np.reshape(posterior_mean, [96, 144])
    posterior_tas_std = np.sqrt(posterior_variance[0, 0])  # This is constant for this model

    return posterior_tas, posterior_tas_std


# External files to download.
EXTERNAL_DEPENDENCIES = {
    "climatebench_webapp.tar.gz": {
        "url": "https://gws-access.jasmin.ac.uk/public/impala/dwatsonparris/climatebench_webapp.tar.gz",
    },
}


if __name__ == "__main__":
    main()
