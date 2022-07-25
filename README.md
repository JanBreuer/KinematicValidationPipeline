# KinematicValidationPipeline
A Validation Pipeline For Markerless Motion Capture Using Data From 3D Cameras And Deep-Learning Libraries


The clinical assessment of certain movement disorders, as well as injuries, is highly subjective which makes the interpretation of such evaluations difficult across groups. While it is possible to capture kinematic data and obtain a quantitative analysis, most clinics lack the resources and experience for such procedures. Newer systems omit the use of reflective markers and apply machine learning to simplify this process. Since these developments do not achieve clinical accuracy yet, more assessment techniques must be established and validated.

The purpose of this work is the facilitation of validating such systems through an open-source graphical user interface. It proposes a modular functioning backend structure creating a pipeline that (1) stores / manages the data, (2) preprocesses the data through interpolation, truncation, and multiple filtering steps, (3) derives multiple parameters used for assessing movement, and (4) estimates the similarity between a new development and a benchmark that is proven to obtain clinical accuracy. Additionally, the software illustrated in this work can visualize the data across several participants and trials. This thesis uses an innovative device  to showcase an exemplary comparison with the VICON system and illustrate the validation of modern 3D-deep-learning marker-less kinematic data. The dataset contains yet unpublished data on a reach and grasp task in healthy adults.

For the wrist joint, an accuracy of 0.976 could be determined in the velocity profile using the software. Less precise results were obtained in the other parameters. Despite limited features and pipeline structures, this project provides a framework for an open-source project within the field and outlines expansions of this toolbox for later projects.

-------

Reaching = VICON files
ReachingPEACK = PEACK files
ReachingValidation = Python files

After downloading, the file directories need to be altered in the ReachingValidation/variableDeclarion.py file.

Then run the ReachingValidation/GUI.py file
