import json
import pickle
from pathlib import Path
from typing import Literal, TypedDict, cast

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt  # type: ignore
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon  # type: ignore
from matplotlib.pyplot import cm  # type: ignore
from matplotlib.transforms import Affine2D  # type: ignore
from numpy import typing as npt
from .roque_cmap import roque, roque_chill

# Define types for the geometry data
GeometryNode = TypedDict(
    "GeometryNode",
    {
        "q": int,
        "r": int,
        "s": int,
        "spiral": int,
        "x": float,
        "y": float,
        "neighbors": list[int],
        "pairs": list[list[int]],
        "edges": list[list[float]],
        "3NN": list[list[int]],
        "4NN": list[list[int]],
    },
)


class GeometryData(TypedDict):
    nodes: dict[int, GeometryNode]


class Geometry:
    n_pixels = 1039

    pixel_numbers = list(range(n_pixels))

    @staticmethod
    def centered_hex_numbers(n: int) -> int:
        if n == 0:
            return 0
        return 3 * (n**2) - 3 * n + 1

    @staticmethod
    def get_size(x: int) -> int:
        size = 0
        c = 0
        while x > c - 1:
            size += 1
            c = Geometry.centered_hex_numbers(size)
        return size - 1

    @staticmethod
    def get_r(x: int, size: int | None = None) -> int:
        if not size:
            size = Geometry.get_size(x)
        c = Geometry.centered_hex_numbers(size)
        c1 = Geometry.centered_hex_numbers(size + 1)
        opposite = (c + c1) / 2

        if (x == c) or (x == opposite):
            return 0

        if (x > c) and (x < opposite):
            sign = -1
            reduced = x
        else:
            sign = 1
            reduced = int(2 * opposite - x)

        dist_c = int(abs(reduced - c))
        dist_opp = int(abs(reduced - opposite))

        if dist_opp < dist_c:
            dist = dist_opp
        else:
            dist = dist_c

        if dist > size:
            return size * sign
        else:
            return dist * sign

    @staticmethod
    def get_q(x: int, size: int | None = None) -> int:
        if not size:
            size = Geometry.get_size(x)

        c1 = Geometry.centered_hex_numbers(size + 1)

        new_x = x + size
        sign = -1
        if x > (c1 - size):
            sign = 1
        return sign * Geometry.get_r(new_x, size)

    @staticmethod
    def get_s(x: int, size: int | None = None) -> int:
        if not size:
            size = Geometry.get_size(x)
        return -Geometry.get_r(x, size) - Geometry.get_q(x, size)

    @staticmethod
    def spiral_to_axial(x: int) -> tuple[int, int, int]:
        size = Geometry.get_size(x)
        c = Geometry.centered_hex_numbers(size)
        c1 = Geometry.centered_hex_numbers(size + 1)
        opposite = (c + c1) / 2

        # find r
        if (x == c) or (x == opposite):
            r = 0

        else:
            if (x > c) and (x < opposite):
                sign = -1
                reduced = x
            else:
                sign = 1
                reduced = int(2 * opposite - x)

            dist_c = int(abs(reduced - c))
            dist_opp = int(abs(reduced - opposite))

            if dist_opp < dist_c:
                dist = dist_opp
            else:
                dist = dist_c

            if dist > size:
                r = size * sign
            else:
                r = dist * sign

        # find q
        new_x = x + size
        sign = -1
        if x > (c1 - size):
            sign = 1
        q = sign * Geometry.get_r(new_x, size)

        # find s
        s = -r - q

        return q, r, s

    @staticmethod
    def pixel_to_axial(p: int) -> tuple[int, int, int, int]:
        offset = {
            -1: 0,
            817: -1,
            833: -2,
            849: -3,
            865: -4,
            881: -5,
            897: -6,
            913: -9,
            926: -14,
            939: -19,
            952: -24,
            965: -29,
            978: -34,
            991: -42,
            999: -53,
            1007: -64,
            1015: -75,
            1023: -86,
            1031: -97,
        }

        if p < 0:
            raise ValueError("pixel value must be positive")
        elif p > 1038:
            raise ValueError("pixel value must be less than 1039")
        elif p < 817:
            q, r, s = Geometry.spiral_to_axial(p)
            return p, q, r, s
        else:
            offset_number = max([x for x in list(offset.keys()) if x < p + 1])
            actual_pixel = p - offset[offset_number]  # -1
            q, r, s = Geometry.spiral_to_axial(actual_pixel)
            return actual_pixel, q, r, s

    @staticmethod
    def lookup_table() -> dict[tuple[int, int], int]:
        magic_geom_lookup = {}
        for i in range(Geometry.n_pixels):
            _, q, r, _ = Geometry.pixel_to_axial(i)
            magic_geom_lookup[(q, r)] = i

        return magic_geom_lookup

    @staticmethod
    def axial_to_pixel(q: int, r: int, s: int | None = None) -> int:
        i = Geometry.lookup_table().get((q, r), -1)

        return i

    @staticmethod
    def cube_to_axial(q: int, r: int, s: int | None = None) -> tuple[int, int]:
        return q, r

    @staticmethod
    def axial_to_cube(
        q: int | float, r: int | float, s: int | float | None = None
    ) -> tuple[int | float, int | float, int | float]:
        q = q
        r = r
        s = -q - r
        return q, r, s

    @staticmethod
    def cube_round(
        q: int | float, r: int | float, s: int | float
    ) -> tuple[int, int, int]:
        round_q = round(q)
        round_r = round(r)
        round_s = round(s)

        q_diff = abs(round_q - q)
        r_diff = abs(round_r - r)
        s_diff = abs(round_s - s)

        if q_diff > r_diff and q_diff > s_diff:
            round_q = -round_r - round_s
        elif r_diff > s_diff:
            round_r = -round_q - round_s
        else:
            round_s = -round_q - round_r

        return round_q, round_r, round_s

    @staticmethod
    def axial_round(
        q: int | float, r: int | float, s: int | float | None = None
    ) -> tuple[int, int]:
        return Geometry.cube_to_axial(
            *Geometry.cube_round(*Geometry.axial_to_cube(q, r, s))
        )

    @staticmethod
    def pixel_to_pointy_hex(
        x: float | int, y: float | int, size: float | int = 1
    ) -> tuple[int, int]:
        q = (np.sqrt(3) / 3 * x - 1.0 / 3 * y) / size
        r = (2.0 / 3 * y) / size
        return Geometry.axial_round(q, r)

    @staticmethod
    def pointy_hex_to_pixel(
        q: int, r: int, s: int | None = None, size: float = 1
    ) -> tuple[float, float]:
        x = size * (np.sqrt(3) * q + np.sqrt(3) / 2 * r)
        y = size * (-3.0 / 2 * r)
        return x, y

    @staticmethod
    def axial_direction(direction: int) -> npt.NDArray[np.float64 | np.int64]:
        axial_direction_vectors = [
            np.array([+1, 0]),
            np.array([+1, -1]),
            np.array([0, -1]),
            np.array([-1, 0]),
            np.array([-1, +1]),
            np.array([0, +1]),
        ]
        return axial_direction_vectors[direction]

    @staticmethod
    def axial_add(
        q: int, r: int, vec: npt.NDArray[np.float64 | np.int64]
    ) -> tuple[int, int]:
        return q + vec[0], r + vec[1]

    @staticmethod
    def axial_neighbor(q: int, r: int, direction: int) -> tuple[int, int]:
        """Direction from 0 to 5, counter-clockwise from +x axis"""
        return Geometry.axial_add(q, r, Geometry.axial_direction(direction))

    @staticmethod
    def get_neighbors_axial(q: int, r: int) -> list[tuple[int, int]]:
        neighbors = []
        for i in range(6):
            neighbors.append(Geometry.axial_neighbor(q, r, i))
        return neighbors

    @staticmethod
    def get_neighbors_pixel(pixel: int, include_empty: bool = False) -> list[int]:
        _, q, r, _ = Geometry.pixel_to_axial(pixel)
        neighbors = Geometry.get_neighbors_axial(q, r)
        pixels = [Geometry.axial_to_pixel(q, r) for q, r in neighbors]
        if not include_empty:
            pixels = [x for x in pixels if x != -1]
        return pixels

    @staticmethod
    def get_pixel_knn(pixel: int, k: int) -> list[tuple[int, ...]] | dict[int, object]:
        if k not in [0, 2, 3, 4]:
            raise ValueError(f"k must be either [0, 2, 3, 4] not {k}")

        if k == 2:
            return [(pixel, n) for n in Geometry.get_neighbors_pixel(pixel)]

        neighbors = Geometry.get_neighbors_pixel(pixel)
        neighbors_neighbors = {n: Geometry.get_neighbors_pixel(n) for n in neighbors}
        neighbor_pairs = []
        neighbor_trios = []
        for n in neighbors:
            this_trio = []
            for nn in neighbors_neighbors[n]:
                if nn in neighbors:
                    this_trio.append(nn)
                    new_pair = tuple(sorted([pixel, n, nn]))
                    if new_pair not in neighbor_pairs:
                        neighbor_pairs.append(new_pair)

                        common = np.intersect1d(
                            neighbors_neighbors[n], neighbors_neighbors[nn]
                        )
                        sorted_commons = tuple(sorted(list(np.append(common, [n, nn]))))
                        if sorted_commons not in neighbor_trios:
                            neighbor_trios.append(sorted_commons)

            sorted_trio = tuple(sorted(this_trio + [pixel, n]))
            if sorted_trio not in neighbor_trios:
                neighbor_trios.append(sorted_trio)

        if k == 3:
            return sorted(neighbor_pairs)

        if k == 4:
            return sorted(neighbor_trios)

        if k == 0:
            return {
                2: [(pixel, n) for n in Geometry.get_neighbors_pixel(pixel)],
                3: sorted(neighbor_pairs),
                4: sorted(neighbor_trios),
            }

        else:
            raise ValueError(f"k must be either [0, 2, 3, 4] not {k}")


class Camera:
    def __init__(self, mode: Literal["pkl", "json"] = "pkl"):
        self._geom: dict[int, GeometryNode]

        if mode == "pkl":
            with open(Path(__file__).with_name("geometry.pkl"), "rb") as f:
                self._geom = pickle.load(f)

        elif mode == "json":
            with open(Path(__file__).with_name("geometry.json"), "r") as f:
                self._geom = cast(dict[int, GeometryNode], json.load(f))

        else:
            raise ValueError("mode must be either 'pkl' or 'json'")

        self.n_pixels = len(self._geom)

    def __getitem__(self, key: int) -> GeometryNode:
        return self._geom[key]

    @property
    def geometry(self) -> dict[int, GeometryNode]:
        return self._geom

    @property
    def NN2(self) -> list[tuple[int, int]]:
        return [
            cast(tuple[int, int], tuple(item))
            for sublist in [p["pairs"] for p in self._geom.values()]
            for item in sublist
        ]

    @property
    def NN3(self) -> list[tuple[int, int, int]]:
        return [
            cast(tuple[int, int, int], tuple(item))
            for sublist in [p["3NN"] for p in self._geom.values()]
            for item in sublist
        ]

    @property
    def NN4(self) -> list[tuple[int, int, int, int]]:
        return [
            cast(tuple[int, int, int, int], tuple(item))
            for sublist in [p["4NN"] for p in self._geom.values()]
            for item in sublist
        ]

    def get_edges(
        self, add_reverse: bool = True, telescope_index: int = 0
    ) -> tuple[list[int], list[int], list[int]]:
        sources, destinations, edge_type = [], [], []

        # 3000 connections between 1039 pixels over 50 time slices
        # node ids are pixel_id * 50 + timeslice

        for pixel_id in range(1039):
            pairs = self.get_pixel_pairs(pixel_id)

            # i.e. create vertical strings first
            for timeslice in range(50):
                node_id = pixel_id * 50 + timeslice

                # connect temporal adjacent pixels
                if timeslice < 49:
                    sources.append(node_id)
                    destinations.append(node_id + 1)
                    edge_type.append(1)  # temporal

                # connect to adjacent pixels
                for pair in pairs:
                    sources.append(node_id)
                    destinations.append(pair[1] * 50 + timeslice)
                    edge_type.append(0)  # spatial

                    if add_reverse:
                        sources.append(pair[1] * 50 + timeslice)
                        destinations.append(node_id)
                        edge_type.append(0)  # spatial

        if telescope_index:
            sources = [x + telescope_index * 1039 * 50 for x in sources]
            destinations = [x + telescope_index * 1039 * 50 for x in destinations]

        return sources, destinations, edge_type

    def get_pixel_pairs(self, pixel: int) -> list[list[int]]:
        return self.geometry[pixel]["pairs"]
    
    def get_pixel_neighbors(self, pixel: int) -> list[int]:
        return self.geometry[pixel]["neighbors"]
    
    def get_pixel_coordinates_xy(self, pixel: int) -> tuple[float, float]:
        return self.geometry[pixel]["x"], self.geometry[pixel]["y"]

    def plot_camera(
        self,
        figsize: int | tuple[int | float, int | float] | None = None,
        text: bool = True,
    ) -> None:
        """Plot a nice image of the camera pixels and their numbering.

        Args:
            figsize (Optional[Union[int, tuple]], optional): Size of the resulting plot. Defaults to (15, 15).
        """

        if isinstance(figsize, int):
            figsize = (figsize, figsize)
        elif isinstance(figsize, tuple):
            figsize = (figsize[0], figsize[1])
        else:
            figsize = (15, 15)

        # change the figsize
        plt.figure(figsize=figsize)
        size = 1

        ax = plt.gca()

        color = cm.prism(np.linspace(0, 1, 20))

        for k, v in self._geom.items():
            dist = Geometry.get_size(v["spiral"])
            ax.add_patch(
                RegularPolygon(
                    (v["x"], v["y"]),
                    6,
                    radius=size,
                    facecolor=color[dist],
                    edgecolor="green",
                    alpha=0.1,
                )
            )
            if text:
                ax.text(v["x"], v["y"], k, ha="center", va="center", fontsize=7)

        plt.xlim(-35, 35)
        plt.ylim(-35, 35)
        plt.title("The MAGIC Camera")
        plt.show()


class Event:
    """A class representing a single event from the MAGIC telescope data.
    
    This class handles the storage and visualization of event data including images,
    timing information, and truth values from both MAGIC telescopes (M1 and M2).

    Attributes:
        _data (pd.Series): The raw data row containing all event information
        _event_number (int): Unique identifier for this event
        _run_number (int): Run number this event belongs to
        _true_energy (float): True energy of the incident particle
        _true_theta (float): True theta angle of incident particle
        _true_phi (float): True phi angle of incident particle
        _telescope_theta (float): Telescope pointing theta angle
        _telescope_phi (float): Telescope pointing phi angle
        _particle_type (str): Type of particle ("gamma" or "hadron")
        _true_first_interaction_height (float): Height of first interaction
        _image_m1 (np.ndarray): Raw image from MAGIC-I telescope
        _image_m2 (np.ndarray): Raw image from MAGIC-II telescope
        _clean_image_m1 (np.ndarray): Cleaned image from MAGIC-I
        _clean_image_m2 (np.ndarray): Cleaned image from MAGIC-II
        _timing_m1 (np.ndarray): Timing information from MAGIC-I
        _timing_m2 (np.ndarray): Timing information from MAGIC-II
    """

    def __init__(self, data_row: pd.Series, particle_type: Literal["gamma", "hadron"]):
        """Initialize an Event object.

        Args:
            data_row (pd.Series): Row of data containing all event information
            particle_type (Literal["gamma", "hadron"]): Type of incident particle
        """
        self._data = data_row
        self._event_number = data_row["event_number"]
        self._run_number = data_row["run_number"]
        self._true_energy = data_row["true_energy"]
        self._true_theta = data_row["true_theta"]
        self._true_phi = data_row["true_phi"]
        self._telescope_theta = data_row["true_telescope_theta"]
        self._telescope_phi = data_row["true_telescope_phi"]
        self._particle_type = particle_type
        self._true_first_interaction_height = data_row["true_first_interaction_height"]
        self._image_m1 = data_row["image_m1"]
        self._image_m2 = data_row["image_m2"]
        self._clean_image_m1 = data_row["clean_image_m1"]
        self._clean_image_m2 = data_row["clean_image_m2"]
        self._timing_m1 = data_row["timing_m1"]
        self._timing_m2 = data_row["timing_m2"]

    @property
    def event_number(self) -> int:
        """Get the event number.

        Returns:
            int: Unique identifier for this event
        """
        return self._event_number

    @property
    def run_number(self) -> int:
        """Get the run number.

        Returns:
            int: Run number this event belongs to
        """
        return self._run_number

    @property
    def truth(self) -> dict[str, float]:
        """Get the true (Monte Carlo) parameters of the event.

        Returns:
            dict[str, float]: Dictionary containing true energy, angles and interaction height
        """
        return {
            "energy": self._true_energy,
            "theta": self._true_theta,
            "phi": self._true_phi,
            "first_interaction_height": self._true_first_interaction_height,
        }

    @property
    def pointing(self) -> dict[str, float]:
        """Get the telescope pointing direction.

        Returns:
            dict[str, float]: Dictionary containing telescope theta and phi angles
        """
        return {
            "theta": self._telescope_theta,
            "phi": self._telescope_phi,
        }

    @property
    def particle_type(self) -> Literal["gamma", "hadron"]:
        """Get the particle type.

        Returns:
            Literal["gamma", "hadron"]: Type of the incident particle
        """
        return self._particle_type

    @property
    def particle_id(self) -> int:
        """Get numerical ID for particle type (0 for gamma, 1 for hadron).

        Returns:
            int: Particle type ID
        """
        return 0 if self._particle_type == "gamma" else 1

    @property
    def data(self) -> dict[str, int | float | np.ndarray]:
        """Get all raw data as dictionary.

        Returns:
            dict[str, int | float | np.ndarray]: Dictionary containing all event data
        """
        return self._data.to_dict()

    def get_image(
        self, telescope_id: Literal[1, 2, "M1", "M2"], cleaned: bool = False
    ) -> np.ndarray:
        """Get the image from specified telescope.

        Args:
            telescope_id (Literal[1, 2, "M1", "M2"]): Which telescope to get image from
            cleaned (bool, optional): Whether to return cleaned image. Defaults to False.

        Returns:
            np.ndarray: Image data from specified telescope
        """
        if cleaned:
            if telescope_id == 1 or telescope_id == "M1":
                return self._clean_image_m1
            else:
                return self._clean_image_m2
        else:
            if telescope_id == 1 or telescope_id == "M1":
                return self._image_m1
            else:
                return self._image_m2

    def get_timing(self, telescope_id: Literal[1, 2, "M1", "M2"]) -> np.ndarray:
        """Get timing information from specified telescope.

        Args:
            telescope_id (Literal[1, 2, "M1", "M2"]): Which telescope to get timing from

        Returns:
            np.ndarray: Timing data from specified telescope
        """
        if telescope_id == 1 or telescope_id == "M1":
            return self._timing_m1
        else:
            return self._timing_m2

    def plot(
        self,
        telescope_id: Literal[1, 2, "M1", "M2"],
        cleaned: bool = False,
        timing: bool = False,
        figsize: int | tuple[int, int] = (8, 8),
        rotate: bool = True,
        simple: bool = False,
        hide_axes: bool = False,
        filename: str | None = None,
        return_ax: bool = False,
        highlight_surviving: bool = False,
        title: str | bool = False,
        transparent: bool = True,
        colormap: str | None = None,
    ) -> None | plt.Axes:
        """Plot the event data as a camera image.

        Args:
            telescope_id (Literal[1, 2, "M1", "M2"]): Which telescope to plot
            cleaned (bool, optional): Whether to plot cleaned image. Defaults to False.
            timing (bool, optional): Whether to plot timing information. Defaults to False.
            figsize (int | tuple[int, int], optional): Figure size. Defaults to (8, 8).
            rotate (bool, optional): Whether to rotate image. Defaults to True.
            simple (bool, optional): Hide axes and colorbar. Defaults to False.
            hide_axes (bool, optional): Hide only axes. Defaults to False.
            filename (str | None, optional): Save plot to file. Defaults to None.
            return_ax (bool, optional): Return axes instead of showing. Defaults to False.
            highlight_surviving (bool, optional): Highlight surviving pixels. Defaults to False.
            title (str | bool, optional): Custom title or auto-generate if True. Defaults to False.
            transparent (bool, optional): Make background transparent. Defaults to True.
            colormap (str | None, optional): Custom colormap name. Defaults to None.

        Returns:
            None | plt.Axes: Matplotlib axes if return_ax is True, else None
        """
        # Get appropriate image data based on timing flag
        if timing:
            image = self.get_timing(telescope_id)

            if cleaned:
                cleaned_image = self.get_image(telescope_id, cleaned=True)
                # set image to zero where cleaned image is zero
                image = np.where(cleaned_image > 0, image, 0)
        else:
            image = self.get_image(telescope_id, cleaned)

        # Get surviving pixels mask if highlighting
        if highlight_surviving:
            cleaned_image = self.get_image(telescope_id, cleaned=True)
            surviving = np.where(cleaned_image > 0, True, False)

        # Set colormap
        if colormap is None or colormap == "roque_chill":
            colormap = roque_chill()
        elif colormap == "roque":
            colormap = roque()

        # Handle figsize input
        if isinstance(figsize, int):
            figsize = (figsize, figsize)
        elif isinstance(figsize, tuple):
            figsize = (figsize[0], figsize[1])

        # Create figure with transparent background
        fig = plt.figure(figsize=figsize)
        if transparent:
            fig.patch.set_alpha(0)  # Make figure background transparent

        ax = fig.add_subplot(111)
        if transparent:
            ax.patch.set_alpha(0)  # Make axes background transparent

        # Create hexagonal patches for each pixel
        patches = []

        for i in range(Geometry.n_pixels):
            _, q, r, _ = Geometry.pixel_to_axial(i)
            x, y = Geometry.pointy_hex_to_pixel(q, r, size=1)

            # Set edge properties based on highlighting
            if highlight_surviving:
                if surviving[i]:
                    edgecolor = (1, 0, 0, 0.3)
                    line_width = 1
                else:
                    edgecolor = (0, 0, 0, 0)
                    line_width = 0
            else:
                edgecolor = (0, 0, 0, 0)
                line_width = 0

            patches.append(
                RegularPolygon(
                    (x, y),
                    6,
                    radius=1,
                    facecolor="none",
                    edgecolor=edgecolor,
                    alpha=0.8,
                    linewidth=line_width,
                )
            )

        # Apply rotation transform if requested
        rotation = Affine2D().rotate_around(0, 0, 1 / 3) if rotate else Affine2D()

        # Create patch collection with all pixels
        collection = PatchCollection(
            patches,
            cmap=colormap,
            alpha=0.8,
            transform=rotation + ax.transData,
            match_original=True if highlight_surviving else False,
        )

        # Set appropriate unit label
        unit = "ns" if timing else "p.e." if cleaned else "ADC Counts"

        # Set up plot elements
        collection.set_array(image)
        collection.set_clim(vmin=0)
        ax.add_collection(collection)
        cb = fig.colorbar(collection, ax=ax, label=unit)
        plt.xlim(-35, 35)
        plt.ylim(-35, 35)
        ax.set_aspect("equal")

        # Handle title
        if title is True:
            title = f"M{telescope_id}{' timing' if timing else ''} - Run {self._run_number} - Event {self._event_number} {self._particle_type}"
        elif title:
            title = title
        else:
            title = None

        if title:
            plt.title(title)

        # Handle display options
        if simple:
            ax.set_axis_off()
            cb.remove()
        elif hide_axes:
            ax.set_axis_off()

        # Handle output
        if filename:
            plt.savefig(filename, transparent=True)  # Add transparent=True for saving
            plt.close()
        elif return_ax:
            return ax
        else:
            plt.show()
