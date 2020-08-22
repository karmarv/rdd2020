# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This demo lets you to explore the Udacity self-driving car image dataset.
# More info: https://github.com/streamlit/demo-self-driving

import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os, urllib, cv2

from model import model 

# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("app.py"))
    elif app_mode == "Run the app":
        readme_text.empty()
        run_the_app()



# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():
    # To make Streamlit fast, st.cache allows us to reuse computation across runs.
    # In this common pattern, we download data from an endpoint only once.
    @st.cache
    def load_metadata():
        """ 
            frame                  , xmin, ymin, xmax, ymax, label
            1478019952686311006.jpg,  254,  153,  269,  166, car
            1478019952686311006.jpg,  468,  128,  487,  199, pedestrian
            1478019953180167674.jpg,  233,  157,  248,  169, car
        """
        for dataset_name, splits_per_dataset in model._PREDEFINED_SPLITS_GRC_MD["rdd2020"].items():
            d = dataset_name.split("_")[1]
            print("[",d,"]\t",dataset_name, "\t", splits_per_dataset)
            annotations = model.load_images_ann_dicts(model.ROADDAMAGE_DATASET, splits_per_dataset)
        return annotations

    # This function uses some Pandas magic to summarize the metadata Dataframe.
    @st.cache
    def create_summary(metadata):
        one_hot_encoded = pd.get_dummies(metadata[["frame", "label","full_file"]], columns=["label"])
        summary = one_hot_encoded.groupby(["frame","full_file"]).sum().rename(columns={
            "label_D00" : "D00",
            "label_D01" : "D01",
            "label_D10" : "D10",
            "label_D11" : "D11",
            "label_D20" : "D20",
            "label_D40" : "D40",
            "label_D43" : "D43",
            "label_D44" : "D44",
            "label_D50" : "D50",
            "label_D0w0": "D0w0"
        })
        return summary

    # An amazing property of st.cached functions is that you can pipe them into
    # one another to form a computation DAG (directed acyclic graph). Streamlit
    # recomputes only whatever subset is required to get the right answer!
    metadata = load_metadata()
    summary = create_summary(metadata)

    # Draw the UI elements to search for objects (pedestrians, cars, etc.)
    selected_frame_index, selected_frame = frame_selector_ui(summary)
    if selected_frame_index == None:
        st.error("No frames fit the criteria. Please select different label or number.")
        return

    # Draw the UI element to select parameters for the YOLO object detector.
    confidence_threshold, overlap_threshold = object_detector_ui()

    # Load the image from S3.
    image_url = os.path.join(selected_frame[1])
    image = load_image(image_url)
    

    # A.) Get the boxes for the objects detected by YOLO by running the YOLO model.
    boxes = model.predict_rdd(image, confidence_threshold)
    draw_image_with_boxes(image, boxes, "Real-time Road Damage Detection",
        "**Faster RCNN Resnet 50 Model** (overlap `%3.1f`) (confidence `%3.1f`) -`%s`" % (overlap_threshold, confidence_threshold, selected_frame[0]))

    # B.) Uncomment these lines to peek at these DataFrames.
    st.write('## Summary', summary[:10], '## Metadata', metadata[:10])

    # C.) Add boxes for objects on the image. These are the boxes for the ground image.
    boxes = metadata[metadata.frame == selected_frame[0]].drop(columns=["frame", "full_file"])
    draw_image_with_boxes(image, boxes, "Ground Truth",
        "**Human-annotated data** (frame `%i`-`%s`)" % (selected_frame_index, selected_frame[0]))



# This sidebar UI is a little search engine to find certain object types.
def frame_selector_ui(summary):
    st.sidebar.markdown("# Frame")

    # The user can pick which type of object to search for.
    object_type = st.sidebar.selectbox("Search for which objects?", summary.columns, 2)

    # The user can select a range for how many of the selected objecgt should be present.
    min_elts, max_elts = st.sidebar.slider("How many %ss (select a range)?" % object_type, 0, 25, [1, 10])
    selected_frames = get_selected_frames(summary, object_type, min_elts, max_elts)
    if len(selected_frames) < 1:
        return None, None
    
    # Choose a frame out of the selected frames.
    selected_frame_index = st.sidebar.slider("Choose a frame (index)", 0, len(selected_frames) - 1, 0)

    # Draw an altair chart in the sidebar with information on the frame.
    objects_per_frame = summary.loc[selected_frames, object_type].reset_index(drop=True).reset_index()
    chart = alt.Chart(objects_per_frame, height=120).mark_area().encode(
        alt.X("index:Q", scale=alt.Scale(nice=False)),
        alt.Y("%s:Q" % object_type))
    selected_frame_df = pd.DataFrame({"selected_frame": [selected_frame_index]})
    vline = alt.Chart(selected_frame_df).mark_rule(color="red").encode(
        alt.X("selected_frame:Q", axis=None)
    )
    st.sidebar.altair_chart(alt.layer(chart, vline))

    selected_frame = selected_frames[selected_frame_index]
    return selected_frame_index, selected_frame

# Select frames based on the selection in the sidebar
@st.cache(hash_funcs={np.ufunc: str})
def get_selected_frames(summary, label, min_elts, max_elts):
    return summary[np.logical_and(summary[label] >= min_elts, summary[label] <= max_elts)].index

# This sidebar UI lets the user select parameters for the YOLO object detector.
def object_detector_ui():
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold

# Draws an image with boxes overlayed to indicate the presence of cars, pedestrians etc.
def draw_image_with_boxes(image, boxes, header, description):
    # Superpose the semi-transparent object detection boxes.    # Colors for the boxes
    LABEL_COLORS = model.RDD_DAMAGE_LABEL_COLORS
    image_with_boxes = image.astype(np.float64)
    if 'scores' in boxes.columns:
        for _, (xmin, ymin, xmax, ymax, label, score) in boxes.iterrows():
            cv2.putText(image_with_boxes, text="{0} ({1:1.2f})".format(label, score), org=(int(xmin+2),int(ymin+10)),
                fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255,255,255),
                thickness=1, lineType=cv2.LINE_AA)
            image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] += LABEL_COLORS[label]
            image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] /= 2
    else:
        for _, (xmin, ymin, xmax, ymax, label) in boxes.iterrows():
            cv2.putText(image_with_boxes, text=label, org=(int(xmin+2),int(ymin+10)),
                fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(255,255,255),
                thickness=1, lineType=cv2.LINE_AA)
            image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] += LABEL_COLORS[label]
            image_with_boxes[int(ymin):int(ymax),int(xmin):int(xmax),:] /= 2

    # Draw the header and image.
    st.subheader(header)
    st.markdown(description)
    st.image(image_with_boxes.astype(np.uint8), use_column_width=True)

# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    with open(path, 'r', encoding="utf8") as myfile:
        return myfile.read()
    return None

# This function loads an image from Streamlit public repo on S3. We use st.cache on this
# function as well, so we can reuse the images across runs.
@st.cache(show_spinner=False)
def load_image(image_fullPath):
    image = cv2.imread(image_fullPath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


if __name__ == "__main__":
    main()
