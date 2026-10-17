"""Separate archived project artifacts from newly generated persona outputs."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PersonaPaths:
    root: Path
    output_dir: Path | None = None

    @property
    def base(self):
        return self.root if self.output_dir is None else self.output_dir

    @property
    def segments(self):
        directory = 'data/processed/segmentation' if self.output_dir is None else 'data'
        return self.base / directory / 'review_segments_all_history.parquet'

    @property
    def results(self):
        return self.base / ('results/user_persona' if self.output_dir is None else 'results')

    @property
    def audit(self):
        return self.results / 'historical_segmentation_audit.json'

    @property
    def manifest(self):
        return self.results / 'analysis_manifest.json'

    @property
    def verification(self):
        return self.results / 'verification.json'

    @property
    def tables(self):
        return self.base / ('reports/tables/user_persona' if self.output_dir is None else 'tables')

    @property
    def figures(self):
        return self.base / ('reports/figures/user_persona' if self.output_dir is None else 'figures')

    @property
    def report(self):
        return self.base / ('reports/sections/complaint_profiles.md' if self.output_dir is None else 'reports/complaint_profiles.md')

    @property
    def readme(self):
        return self.base / ('docs/user_persona.md' if self.output_dir is None else 'README.md')

    def create(self):
        for path in [self.segments.parent,self.results,self.tables,self.figures,self.report.parent]:
            path.mkdir(parents=True,exist_ok=True)
