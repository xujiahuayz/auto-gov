"""
TODO: to be cleaned up
script to generate a table of posts in Aave Governance Forum
"""

from os import path
import matplotlib.pyplot as plt
import time


from dateutil.parser import parse

import requests
import pandas as pd
from bs4 import BeautifulSoup
from market_env.constants import DATA_PATH, FIGURE_PATH


def get_soup(actualurl: str) -> BeautifulSoup:
    getpage = requests.get(url=actualurl)
    getpage_soup = BeautifulSoup(getpage.text, "html.parser")
    return getpage_soup


def getcontent(getpage_soup: BeautifulSoup) -> dict:
    categories = [
        x.get_text()
        for x in getpage_soup.findAll("span", attrs={"class": "category-name"})
    ]

    posts = pd.DataFrame()

    posts["datetime"] = [
        x.attrs["datetime"]
        for x in getpage_soup.findAll("time", attrs={"class": "post-time"})
    ]
    posts["creator"] = [
        x.get_text().strip("\n")
        for x in getpage_soup.findAll("span", attrs={"class": "creator"})
    ]

    return {
        # 'topCategory': topCategory,
        "categories": categories,
        "posts": posts,
    }


def post_table(total_page_number: int) -> pd.DataFrame:
    root_url = "https://governance.aave.com/latest?page="

    all_titles = pd.DataFrame()

    for page_number in range(total_page_number):
        print(page_number)

        links = get_soup(root_url + str(page_number))

        titles = pd.DataFrame()
        titles["title"] = [
            x.attrs["content"]
            for x in links.findAll("meta", attrs={"itemprop": "name"})
        ]
        titles["category"] = [
            x.get_text()
            for x in links.findAll("span", attrs={"class": "category-name"})
        ]
        titles["noPosts"] = [
            x.get_text() for x in links.findAll("span", attrs={"class": "posts"})
        ]
        titles["noViews"] = [
            x.get_text() for x in links.findAll("span", attrs={"class": "views"})
        ]
        titles["url"] = [
            x.get("href")
            for x in links.findAll("a", class_="title raw-link raw-topic-link")
        ]

        categories_posts = []

        # need to repeat retrieving with intervals when failing once
        for x in titles["url"]:
            categories_posts_page = getcontent(get_soup(x))
            while categories_posts_page["categories"] == []:
                print(x)
                time.sleep(10)
                categories_posts_page = getcontent(get_soup(x))
            categories_posts.append(categories_posts_page)

        new_titles = pd.concat([titles, pd.DataFrame(categories_posts)], axis=1)

        all_titles = all_titles.append(new_titles, ignore_index=True)
    return all_titles


if __name__ == "__main__":
    POST_PATH = path.join(DATA_PATH, "AaveGovernance.pkl")

    alltitles = post_table(total_page_number=37)
    alltitles.to_pickle(POST_PATH)

    alltitles = pd.read_pickle(POST_PATH)

    # fetch top-level category
    alltitles["topCategory"] = [x[0] for x in alltitles["categories"]]

    # fetch timestamp of initial posts
    alltitles["timePosted"] = [parse(x["datetime"][0], "") for x in alltitles["posts"]]

    topCategory_counts = alltitles.value_counts("topCategory")
    offset = 0

    ytickposition = [offset + 1]
    yticklabel = ["Sub-categories"]

    ytickposition_right = [offset + 1]
    yticklabel_right = ["Main categories"]

    fig, ax = plt.subplots()
    ax.tick_params(axis="y", which="both", length=0)

    for cat in topCategory_counts.index:
        series = alltitles.loc[alltitles["topCategory"] == cat]
        category_count = series.value_counts("category")
        offset -= 0.25
        for subcat in category_count.index:
            offset -= 1
            to_append = series["timePosted"][series["category"] == subcat]

            plt.eventplot(
                to_append,
                lineoffsets=offset,
                linelengths=0.5,
                linewidths=0.75,
                colors="k",
            )

            ytickposition.append(offset)
            yticklabel.append(subcat + " " + "{: >3d}".format(category_count[subcat]))

        plt.axhline(y=offset - 0.75, color="grey", lw=0.8)

        ytickposition_right.append(offset + 0.5 * len(category_count) - 0.5)
        yticklabel_right.append(cat + "\n" + "{: >3d}".format(topCategory_counts[cat]))

    plt.yticks(ytickposition, yticklabel)
    plt.xticks(rotation=90)

    ax_right = ax.twinx()
    ax_right.set_ylim(ax.get_ylim())
    ax_right.tick_params(axis="y", which="both", length=0)
    plt.yticks(ytickposition_right, yticklabel_right)
    plt.ylabel(
        "Total\n" + "{: >3d}".format(topCategory_counts.sum()),
        loc="bottom",
        labelpad=-80,
        rotation="horizontal",
        weight="bold",
    )
    plt.tight_layout()

    plt.savefig(FIGURE_PATH / "aave_desc_tbl.pdf")
