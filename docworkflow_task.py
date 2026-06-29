""" Bridge between RTK (Release the Krakens) orchestration and DocWorkflow ML tasks.

RTK orchestrates large-scale HTR pipelines from IIIF manifests: it batches manifests
to control disk usage, downloads images, and chains processing steps through a
``check()`` / ``process()`` idempotence contract where each step's ``output_files``
feed the next step's ``input_files``.

DocWorkflow provides the ML backend (layout -> line -> HTR) as Python ``BaseTask``
instances (``YoloLayout``, ``KrakenLine``, ``VLMLineHTR``, ...). A DocWorkflow task
exposes ``predict(data_path, output_dir)`` where ``data_path`` is a *folder* it scans
with ``discover_dataset_structure()``.

``DocWorkflowTask`` wraps a DocWorkflow ``BaseTask`` so it behaves like an RTK
``Task``: it reconciles RTK's per-file lists with DocWorkflow's per-folder ``predict``
by grouping inputs by their parent directory and calling ``predict`` once per group.

Interface notes (verified against docworkflow/src):
  * ``predict(data_path, output_dir, save_image=True)`` discovers input files in
    ``data_path`` by extension (images for layout/line, ``*.xml`` for VLM line HTR),
    lazily calls ``self.load()`` once (the model persists on the instance), and writes
    one ALTO XML per input named ``<stem>.xml`` into ``output_dir``.
  * DocWorkflow stores only the image *basename* in the ALTO ``<fileName>`` and the
    downstream line/HTR tasks resolve the image *next to* the ALTO file. So images
    must travel alongside their XML between stages: keep ``save_image=True``.
  * Passing a single manuscript folder (images directly inside) yields a ``flat``
    structure, so output lands directly in the ``output_dir`` we pass -- we point that
    at ``<output_dir>/<manuscript-folder-name>`` to preserve the per-manuscript layout.
"""

import os
from pathlib import Path
from collections import defaultdict
from typing import List, Optional

import tqdm
import lxml.etree as ET

from rtk.task import Task


def _alto_has_content(file_path: str) -> bool:
    """ Returns True if the ALTO file parses and contains at least one String CONTENT. """
    try:
        root = ET.parse(file_path).getroot()
    except (ET.ParseError, OSError):
        return False
    ns = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}
    return any(s.get("CONTENT") for s in root.findall(".//alto:String", ns))


class DocWorkflowTask(Task):
    """ Runs a DocWorkflow ``BaseTask`` (layout, line or HTR) as an RTK task.

    :param input_files: List of input file paths. For layout these are JPGs; for line
        and HTR they are typically the ALTO XML produced by the previous stage. Only the
        files' *parent directories* matter -- DocWorkflow re-discovers the actual inputs
        (images or XML, depending on the task) inside each folder.
    :param task_instance: An already-constructed DocWorkflow ``BaseTask`` instance
        (e.g. ``YoloLayout``, ``KrakenLine``, ``VLMLineHTR``). Build it **once** outside
        the manifest batch loop so its model loads a single time and is reused across
        batches.
    :param output_dir: Base directory for this stage's ALTO XML output. Each input's
        parent-folder name is preserved as a sub-directory, mirroring RTK's
        per-manuscript layout: ``<output_dir>/<manuscript>/<stem>.xml``.
    :param save_image: Copy the source image next to the produced ALTO XML. Required so
        the image travels with the XML to the next stage (DocWorkflow resolves images
        relative to the ALTO file). Default ``True``.
    :param preload: Load the DocWorkflow model in ``__init__`` instead of lazily on the
        first ``predict``. Default ``False`` (``predict`` loads it on first use anyway).
    :param check_content: In ``check()``, treat an output as done only if the ALTO file
        actually contains transcribed text. Useful for the HTR stage. Default ``False``.
    """

    def __init__(
            self,
            input_files: List[str],
            *args,
            task_instance,
            output_dir: str,
            save_image: bool = True,
            preload: bool = False,
            check_content: bool = False,
            **kwargs):
        super(DocWorkflowTask, self).__init__(input_files=input_files, *args, **kwargs)
        self.task = task_instance
        self.output_dir: str = str(output_dir)
        self.save_image: bool = save_image
        self.check_content: bool = check_content
        self._output_files: List[str] = []
        if preload:
            self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        """ Load the DocWorkflow model once (idempotent). """
        if getattr(self.task, "model", None) is None:
            self.task.load()

    def output_path_for(self, input_file: str) -> str:
        """ Map an input file to the ALTO XML DocWorkflow will produce for it.

        DocWorkflow names outputs ``<stem>.xml``; we preserve the parent (manuscript)
        folder under ``output_dir``.
        """
        p = Path(input_file)
        return os.path.join(self.output_dir, p.parent.name, p.stem + ".xml")

    @property
    def output_files(self) -> List[str]:
        """ ALTO XML paths that exist on disk, one per input, in input order. """
        seen = set()
        ordered = []
        for inp in self.input_files:
            out = self.output_path_for(inp)
            if out not in seen and os.path.exists(out):
                seen.add(out)
                ordered.append(out)
        return ordered

    def check(self) -> bool:
        """ Mark inputs whose ALTO XML output already exists (RTK idempotence). """
        all_done: bool = True
        self._output_files = []
        for inp in tqdm.tqdm(
                self.input_files,
                desc="[Subtask] Checking prior processed documents",
                total=len(self.input_files)):
            out = self.output_path_for(inp)
            done = os.path.exists(out) and (not self.check_content or _alto_has_content(out))
            self._checked_files[inp] = done
            if done:
                self._output_files.append(out)
            else:
                all_done = False
        return all_done

    def _process(self, inputs: List[str]) -> bool:
        """ Run DocWorkflow ``predict`` once per parent directory of the inputs.

        RTK hands us a flat list of files; DocWorkflow wants a folder. Since RTK lays
        images out per manuscript, we group by parent directory and call ``predict``
        on each folder, directing output to ``<output_dir>/<folder-name>``.
        """
        self._ensure_loaded()

        groups = defaultdict(list)
        for inp in inputs:
            groups[str(Path(inp).parent)].append(inp)

        for data_path, files in tqdm.tqdm(
                groups.items(),
                desc=f"[Subtask] DocWorkflow {getattr(self.task, 'name', 'task')}",
                total=len(groups)):
            out_subdir = os.path.join(self.output_dir, Path(data_path).name)
            os.makedirs(out_subdir, exist_ok=True)
            try:
                self.task.predict(
                    data_path=data_path,
                    output_dir=out_subdir,
                    save_image=self.save_image,
                )
            except Exception as e:
                print(f"  Error processing {data_path}: {e}")
                import traceback
                traceback.print_exc()

        self._output_files = self.output_files
        return True
