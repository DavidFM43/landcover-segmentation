
It is important to have a good understanding of the data because:
- The model is a compressed version of the dataset. If we have a good understanding of the dataset, we can better evaluate errors in predictions and provide possible reasons for why they are occurring. If the network is producing something inconsistent with what we have seen in the data, something is wrong.
To gain a good understanding of the dataset, we can ask ourselves the following questions:
- What factors do we consider when classifying an image?
- Is a lot of global context needed, or is local information sufficient?
- How important is the resolution of the images? Is such a high level of detail necessary for successful image classification?
- How much variability is there in the data, and what form does this variation take?
- What variability is irrelevant and can be eliminated during pre-processing?
- Does the spatial position in the images matter?
- How noisy are the labels?

#### Insights

- The **agriculture** class is easy to recognize due to the rectangular shapes formed by the crops on the land. The colors are green-yellow.
- The **urban** class is easy to recognize as it consists of small, clustered houses or buildings, often white in color.
- The **water** class can be recognized by the fact that the texture is very uniform with a color close to blue, although not always. Sometimes it can be more yellowish or greenish.
- The **rangeland** class is more difficult to recognize. An important factor is the color, similar to the **agriculture** class but without the rectangles. Sometimes it has a few trees, but not too many and not clustered together. It is generally difficult to distinguish the boundaries and can be confused with the **barrenland** class. Colors sometimes help, but other times they can be distracting.
- The **forest** class is complicated because most of the time it is easy to recognize trees, but not every group of trees is classified as a forest. In most cases, a certain level of clustering of trees is required, but sometimes this criterion seems to be relaxed.
- In my opinion, the most difficult class to classify is **barrenland**. The most important aspect seems to be a color that is different from green, but sometimes it can be brown.

Overall, the annotations seem ambiguous, both in terms of filling and borders. In many cases, annotations are not consistent within a given class. There are also many imprecise borders and occasional missing annotations.