from pathlib import Path
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk



# Paths


PROJECT_ROOT = Path(
    "/path/to/your/MousePose_Project"
)

ANNOTATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
)

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "dataset_split"
)



# Label options


POSES = [
    "Curled",
    "Crouching",
    "Walking",
    "Rearing",
    "Stretching",
    "Hanging",
    "Upside-down",
    "Drinking/Eating",
    "Other",
]

ORIENTATIONS = [
    "side",
    "front",
    "back",
    "angled",
]

VISIBILITY = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
]

OCCLUSION = [
    "None",
    "Mouse",
    "Water/Food box",
    "Paper roll",
    "Boundary",
]

MOUSE_OCCLUSION = [
    "Near",
    "Partial overlap",
    "Full overlap",
]

USABLE_OPTIONS = [
    "yes",
    "no",
]

MOUSE_FIELDS = [
    "orientation",
    "pose",
    "key_point_visibility",
    "occlusion_type",
    "mouse_occlusion",
]



# Select split


choice = input(
    """
Choose split:

0 = train
1 = val
2 = test

Input: """
).strip()

split_mapping = {
    "0": "train",
    "1": "val",
    "2": "test",
}

if choice not in split_mapping:
    raise ValueError(
        "Invalid split. Please enter 0, 1 or 2."
    )

split = split_mapping[choice]

csv_path = (
        ANNOTATION_DIR
        / f"{split}_metadata.csv"
    )

if not csv_path.exists():
    raise FileNotFoundError(
        f"Metadata CSV does not exist:\n{csv_path}\n\n"
        "Run 07_create_metadata_csv.py first."
    )

df = pd.read_csv(
    csv_path,
    dtype=str,
    keep_default_na=False,
)

if len(df) == 0:
    raise ValueError(
        f"No frames were found in {csv_path}"
    )

print(f"Loaded {split}: {len(df)} frames")



# Validate CSV columns

REQUIRED_COLUMNS = [
    "split",
    "video",
    "image",
    "mouse_count",
    "usable",
]

for mouse_id in range(1, 4):
    for field in MOUSE_FIELDS:
        REQUIRED_COLUMNS.append(
            f"mouse_{mouse_id}_{field}"
        )

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    missing_text = "\n".join(
        missing_columns
    )

    raise ValueError(
        "The metadata CSV is missing these columns:\n\n"
        f"{missing_text}\n\n"
        "Please update and rerun "
        "07_create_metadata_csv.py first."
    )


# GUI

class AnnotationTool:

    def __init__(self, root: tk.Tk):

        self.root = root

        self.root.title(
            f"Mouse Pose Annotation - {split}"
        )

        self.root.minsize(
            1100,
            750,
        )

        self.index = 0
        self.photo = None

        # Structure:
        #
        # {
        #     1: {
        #         "orientation": widget,
        #         "pose": widget,
        #         ...
        #     },
        #     2: {...},
        #     3: {...},
        # }
        self.mouse_widgets = {}

        self.build_layout()
        self.load_image()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.on_close,
        )


    # Build main layout

    def build_layout(self):

        self.root.columnconfigure(
            0,
            weight=1,
        )

        self.root.columnconfigure(
            1,
            weight=0,
        )

        self.root.rowconfigure(
            0,
            weight=1,
        )

        
        # Left panel: image

        self.image_frame = ttk.Frame(
            self.root,
            padding=10,
        )

        self.image_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        self.image_frame.columnconfigure(
            0,
            weight=1,
        )

        self.image_frame.rowconfigure(
            0,
            weight=1,
        )

        self.image_label = ttk.Label(
            self.image_frame,
            anchor="center",
        )

        self.image_label.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        self.info_label = ttk.Label(
            self.image_frame,
            text="",
            justify="center",
            anchor="center",
        )

        self.info_label.grid(
            row=1,
            column=0,
            pady=(10, 0),
        )

        # Right panel: annotation controls

        self.control_frame = ttk.Frame(
            self.root,
            padding=15,
        )

        self.control_frame.grid(
            row=0,
            column=1,
            sticky="ns",
        )

        title = ttk.Label(
            self.control_frame,
            text="Frame Annotation",
            font=("Arial", 16, "bold"),
        )

        title.pack(
            pady=(0, 15),
        )

        # Mouse count

        mouse_count_frame = ttk.LabelFrame(
            self.control_frame,
            text="Number of mice to annotate",
            padding=10,
        )

        mouse_count_frame.pack(
            fill="x",
            pady=(0, 10),
        )

        self.mouse_count_box = ttk.Combobox(
            mouse_count_frame,
            values=["1", "2", "3"],
            state="readonly",
            width=24,
        )

        self.mouse_count_box.pack(
            fill="x",
        )

        self.mouse_count_box.bind(
            "<<ComboboxSelected>>",
            self.on_mouse_count_changed,
        )

        # Scrollable mouse annotation area

        mice_outer_frame = ttk.Frame(
            self.control_frame
        )

        mice_outer_frame.pack(
            fill="both",
            expand=True,
        )

        self.mice_canvas = tk.Canvas(
            mice_outer_frame,
            width=370,
            height=500,
            highlightthickness=0,
        )

        self.mice_scrollbar = ttk.Scrollbar(
            mice_outer_frame,
            orient="vertical",
            command=self.mice_canvas.yview,
        )

        self.mice_canvas.configure(
            yscrollcommand=self.mice_scrollbar.set
        )

        self.mice_scrollbar.pack(
            side="right",
            fill="y",
        )

        self.mice_canvas.pack(
            side="left",
            fill="both",
            expand=True,
        )

        self.mice_container = ttk.Frame(
            self.mice_canvas
        )

        self.mice_canvas_window = (
            self.mice_canvas.create_window(
                (0, 0),
                window=self.mice_container,
                anchor="nw",
            )
        )

        self.mice_container.bind(
            "<Configure>",
            self.update_scroll_region,
        )

        self.mice_canvas.bind(
            "<Configure>",
            self.resize_mice_container,
        )

        # Frame usability

        self.usable_frame = ttk.LabelFrame(
            self.control_frame,
            text="Frame usability",
            padding=10,
        )

        self.usable_frame.pack(
            fill="x",
            pady=(10, 0),
        )

        self.usable_box = ttk.Combobox(
            self.usable_frame,
            values=USABLE_OPTIONS,
            state="readonly",
            width=24,
        )

        self.usable_box.pack(
            fill="x",
        )

        # Buttons

        button_frame = ttk.Frame(
            self.control_frame
        )

        button_frame.pack(
            fill="x",
            pady=(15, 0),
        )

        self.previous_button = ttk.Button(
            button_frame,
            text="Previous",
            command=self.previous,
        )

        self.previous_button.pack(
            side="left",
            padx=(0, 5),
        )

        self.save_button = ttk.Button(
            button_frame,
            text="Save",
            command=self.save_current,
        )

        self.save_button.pack(
            side="left",
            padx=5,
        )

        self.next_button = ttk.Button(
            button_frame,
            text="Save & Next",
            command=self.save_next,
        )

        self.next_button.pack(
            side="left",
            padx=(5, 0),
        )


    # Scrollable panel helpers

    def update_scroll_region(
        self,
        event=None,
    ):

        self.mice_canvas.configure(
            scrollregion=(
                self.mice_canvas.bbox("all")
            )
        )


    def resize_mice_container(
        self,
        event,
    ):

        self.mice_canvas.itemconfigure(
            self.mice_canvas_window,
            width=event.width,
        )



    # Mouse annotation panels

    def on_mouse_count_changed(
        self,
        event=None,
    ):

        mouse_count_text = (
            self.mouse_count_box.get()
        )

        if mouse_count_text not in {
            "1",
            "2",
            "3",
        }:
            return

        self.create_mouse_panels(
            int(mouse_count_text)
        )


    def create_mouse_panels(
        self,
        mouse_count: int,
    ):

        # Remove all previous mouse panels.
        for child in (
            self.mice_container.winfo_children()
        ):
            child.destroy()

        self.mouse_widgets = {}

        for mouse_id in range(
            1,
            mouse_count + 1,
        ):

            panel = ttk.LabelFrame(
                self.mice_container,
                text=f"Mouse {mouse_id}",
                padding=10,
            )

            panel.pack(
                fill="x",
                pady=5,
                padx=(0, 5),
            )

            widgets = {}

            widgets["orientation"] = (
                self.create_combobox_row(
                    parent=panel,
                    label="Orientation",
                    values=ORIENTATIONS,
                    row=0,
                )
            )

            widgets["pose"] = (
                self.create_combobox_row(
                    parent=panel,
                    label="Pose",
                    values=POSES,
                    row=1,
                )
            )

            widgets["key_point_visibility"] = (
                self.create_combobox_row(
                    parent=panel,
                    label="Key-point visibility",
                    values=VISIBILITY,
                    row=2,
                )
            )

            widgets["occlusion_type"] = (
                self.create_combobox_row(
                    parent=panel,
                    label="Occlusion type",
                    values=OCCLUSION,
                    row=3,
                )
            )

            widgets["mouse_occlusion"] = (
                self.create_combobox_row(
                    parent=panel,
                    label="Mouse occlusion",
                    values=MOUSE_OCCLUSION,
                    row=4,
                    state="disabled",
                )
            )

            widgets["occlusion_type"].bind(
                "<<ComboboxSelected>>",
                lambda event,
                current_mouse_id=mouse_id:
                self.update_mouse_occlusion(
                    current_mouse_id
                ),
            )

            self.mouse_widgets[
                mouse_id
            ] = widgets

        self.update_scroll_region()


    def create_combobox_row(
        self,
        parent,
        label: str,
        values: list[str],
        row: int,
        state: str = "readonly",
    ) -> ttk.Combobox:

        ttk.Label(
            parent,
            text=label,
        ).grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 10),
            pady=4,
        )

        box = ttk.Combobox(
            parent,
            values=values,
            state=state,
            width=22,
        )

        box.grid(
            row=row,
            column=1,
            sticky="ew",
            pady=4,
        )

        parent.columnconfigure(
            1,
            weight=1,
        )

        return box


    def update_mouse_occlusion(
        self,
        mouse_id: int,
    ):

        if mouse_id not in self.mouse_widgets:
            return

        widgets = self.mouse_widgets[
            mouse_id
        ]

        occlusion_type = (
            widgets[
                "occlusion_type"
            ].get()
        )

        mouse_occlusion_box = (
            widgets[
                "mouse_occlusion"
            ]
        )

        if occlusion_type == "Mouse":

            mouse_occlusion_box.config(
                state="readonly"
            )

            if (
                mouse_occlusion_box.get()
                not in MOUSE_OCCLUSION
            ):
                mouse_occlusion_box.set("")

        else:

            mouse_occlusion_box.set("NA")

            mouse_occlusion_box.config(
                state="disabled"
            )


    # Image handling

    def get_image_path(self) -> Path:

        row = df.iloc[
            self.index
        ]

        

        return (
                DATASET_DIR
                / split
                / row["video"]
                / row["image"]
            )


    def load_image(self):

        image_path = (
            self.get_image_path()
        )

        if not image_path.exists():

            messagebox.showerror(
                "Image not found",
                f"Cannot find image:\n\n"
                f"{image_path}",
            )

            return

        try:

            with Image.open(
                image_path
            ) as image:

                image = image.convert(
                    "RGB"
                )

                image.thumbnail(
                    (850, 750),
                    Image.Resampling.LANCZOS,
                )

                self.photo = (
                    ImageTk.PhotoImage(
                        image.copy()
                    )
                )

        except Exception as error:

            messagebox.showerror(
                "Image loading error",
                f"Could not open:\n"
                f"{image_path}\n\n"
                f"{error}",
            )

            return

        self.image_label.config(
            image=self.photo
        )

        row = df.iloc[
            self.index
        ]

        self.info_label.config(
            text=(
                f"Frame "
                f"{self.index + 1} / {len(df)}\n"
                f"{row['video']} / {row['image']}"
            )
        )

        self.load_previous_labels()
        self.update_navigation_buttons()


    # Load labels from CSV

    def load_previous_labels(self):

        row = df.iloc[
            self.index
        ]

        mouse_count = str(
            row.get(
                "mouse_count",
                "",
            )
        ).strip()

        if mouse_count not in {
            "1",
            "2",
            "3",
        }:

            self.mouse_count_box.set("")
            self.create_mouse_panels(0)

        else:

            self.mouse_count_box.set(
                mouse_count
            )

            self.create_mouse_panels(
                int(mouse_count)
            )

            for mouse_id in range(
                1,
                int(mouse_count) + 1,
            ):

                widgets = (
                    self.mouse_widgets[
                        mouse_id
                    ]
                )

                for field, widget in (
                    widgets.items()
                ):

                    column = (
                        f"mouse_"
                        f"{mouse_id}_"
                        f"{field}"
                    )

                    value = str(
                        row.get(
                            column,
                            "",
                        )
                    ).strip()

                    if (
                        field
                        == "mouse_occlusion"
                    ):

                        if value in {
                            "",
                            "NA",
                        }:

                            widget.set("NA")

                            widget.config(
                                state="disabled"
                            )

                        else:

                            widget.config(
                                state="readonly"
                            )

                            widget.set(
                                value
                            )

                    else:

                        widget.config(
                            state="readonly"
                        )

                        widget.set(
                            value
                        )

                self.update_mouse_occlusion(
                    mouse_id
                )

        usable_value = str(
            row.get(
                "usable",
                "",
            )
        ).strip()

        self.usable_box.set(
            usable_value
        )


    # Validation

    def validate_current_labels(
        self,
    ) -> bool:

        mouse_count_text = (
            self.mouse_count_box.get()
        )

        if mouse_count_text not in {
            "1",
            "2",
            "3",
        }:

            messagebox.showwarning(
                "Missing mouse count",
                "Please select the number "
                "of mice to annotate.",
            )

            return False

        if (
            self.usable_box.get()
            not in USABLE_OPTIONS
        ):

            messagebox.showwarning(
                "Missing usability",
                "Please select whether "
                "the frame is usable.",
            )

            return False

        # If unusable, detailed mouse labels
        # are not required.
        if (
            self.usable_box.get()
            == "no"
        ):
            return True

        mouse_count = int(
            mouse_count_text
        )

        for mouse_id in range(
            1,
            mouse_count + 1,
        ):

            widgets = (
                self.mouse_widgets[
                    mouse_id
                ]
            )

            required_fields = [
                "orientation",
                "pose",
                "key_point_visibility",
                "occlusion_type",
            ]

            for field in required_fields:

                value = (
                    widgets[
                        field
                    ].get().strip()
                )

                if value == "":

                    messagebox.showwarning(
                        "Missing label",
                        f"Mouse {mouse_id}: "
                        f"please select "
                        f"{field}.",
                    )

                    return False

            if (
                widgets[
                    "occlusion_type"
                ].get()
                == "Mouse"
                and
                widgets[
                    "mouse_occlusion"
                ].get()
                not in MOUSE_OCCLUSION
            ):

                messagebox.showwarning(
                    "Missing mouse occlusion",
                    f"Mouse {mouse_id}: "
                    "please select the "
                    "mouse occlusion level.",
                )

                return False

        return True


    # Save labels

    def save_current(self):

        if not self.validate_current_labels():
            return False

        mouse_count_text = (
            self.mouse_count_box.get()
        )

        mouse_count = int(
            mouse_count_text
        )

        df.at[
            self.index,
            "mouse_count",
        ] = mouse_count_text

        df.at[
            self.index,
            "usable",
        ] = self.usable_box.get()

        # Clear all existing mouse labels.
        #
        # This prevents old Mouse 2 or Mouse 3
        # labels remaining when mouse_count is reduced.
        for mouse_id in range(
            1,
            4,
        ):

            for field in MOUSE_FIELDS:

                column = (
                    f"mouse_"
                    f"{mouse_id}_"
                    f"{field}"
                )

                df.at[
                    self.index,
                    column,
                ] = ""

        # Save labels for visible panels.
        for mouse_id in range(
            1,
            mouse_count + 1,
        ):

            widgets = (
                self.mouse_widgets[
                    mouse_id
                ]
            )

            for field, widget in (
                widgets.items()
            ):

                column = (
                    f"mouse_"
                    f"{mouse_id}_"
                    f"{field}"
                )

                value = (
                    widget.get().strip()
                )

                if (
                    field
                    == "mouse_occlusion"
                ):

                    occlusion_type = (
                        widgets[
                            "occlusion_type"
                        ].get()
                    )

                    if (
                        occlusion_type
                        != "Mouse"
                    ):
                        value = "NA"

                df.at[
                    self.index,
                    column,
                ] = value

        try:

            df.to_csv(
                csv_path,
                index=False,
            )

        except Exception as error:

            messagebox.showerror(
                "Save error",
                f"Could not save CSV:\n\n"
                f"{csv_path}\n\n"
                f"{error}",
            )

            return False

        print(
            f"Saved frame "
            f"{self.index + 1}/{len(df)}"
        )

        return True


    # Navigation
    
    def save_next(self):

        if not self.save_current():
            return

        if self.index < len(df) - 1:

            self.index += 1
            self.load_image()

        else:

            messagebox.showinfo(
                "Finished",
                f"All {split} frames "
                "have been reviewed.",
            )


    def previous(self):

        # Save current frame before moving back.
        if not self.save_current():
            return

        if self.index > 0:

            self.index -= 1
            self.load_image()


    def update_navigation_buttons(self):

        if self.index == 0:

            self.previous_button.config(
                state="disabled"
            )

        else:

            self.previous_button.config(
                state="normal"
            )

        if self.index == len(df) - 1:

            self.next_button.config(
                text="Save & Finish"
            )

        else:

            self.next_button.config(
                text="Save & Next"
            )



    # Close window

    def on_close(self):

        answer = messagebox.askyesnocancel(
            "Close annotation tool",
            "Save the current frame "
            "before closing?",
        )

        if answer is None:
            return

        if answer:

            if not self.save_current():
                return

        self.root.destroy()



# Run


def main():

    root = tk.Tk()

    AnnotationTool(root)

    root.mainloop()


if __name__ == "__main__":
    main()