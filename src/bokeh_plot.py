import bokeh
import bokeh.plotting
from bokeh.io import curdoc
from bokeh.palettes import Category10

import pandas

if "bokeh_app" in __name__ or __name__ == "__main__":

    print(__name__)
    df_plot = pandas.read_pickle("data/umap/df_plot.pkl")

    tooltips = """
    <div>
        <div>
            <img
                src="file://@url" height="200" alt="@url" width="200"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
        </div>
        <div>
            <span style="font-size: 17px;">Class :</span>
            <span style="font-size: 17px; font-weight: bold;"</span>
            <span style="font-size: 15px; color: #966;">@class</span>
        </div>
        <div>
            <span style="font-size: 15px; color: #696;">@type</span>
        </div>
    </div>
    """

    plot = bokeh.plotting.figure(width=1000, height=1000, tooltips=tooltips)

    mapper = bokeh.transform.linear_cmap(field_name='class_color', palette=Category10[6] ,low=0 ,high=len(df_plot["class"].unique()))

    plot.circle(
        source=df_plot[df_plot["type"] == "train"],color=mapper,
    )

    plot.square(
        source=df_plot[df_plot["type"] == "found"],color=mapper,

    )

    plot.triangle(
        source=df_plot[df_plot["type"] == "miss"],color=mapper,
    )

    if __name__ == "__main__":
        bokeh.io.show(plot)
    else:
        curdoc.title = "Interactive plot of feature spaces"
        curdoc().add_root(plot)
