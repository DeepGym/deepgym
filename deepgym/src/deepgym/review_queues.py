"""Review queues: items handed out to expert reviewers, each reviewed by ``overlap`` of them.

A team puts items in a :class:`Queue`: runs, episodes or traces, or one step of one, each
named by a :class:`deepgym.labels.LabelTarget`. A reviewer asks for the next item, which
reserves it for them for ``reservation_seconds``; a reservation that runs out frees the item
for someone else. Submitting a review stores its labels in the label store, as the
reviewer's, with ``group_id`` set to the reservation's id, and counts one review towards the
item's ``overlap``. No reviewer reviews an item twice. In a blind queue a reviewer sees no
label on the item's kind and id until they have submitted their own review of it (see
:class:`ReviewView`), and the review's labels are marked ``blind`` only if no queue response
showed the reviewer someone else's label on that recording and they had no turn at it in a
queue that isn't blind. An episode recorded from a trace is on that trace's recording.

The console keeps queues in its database (:class:`deepgym.console.queue_store.QueueStore`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    model_validator,
)

from deepgym.labels import SQLITE_MAX_INT, Label, LabelTarget, LabelValue, Name, Text

ReviewStatus = Literal['reserved', 'submitted', 'released', 'skipped', 'expired']

MAX_RESERVATION_SECONDS = 10**9
"""About 31 years: the longest reservation, so that its expiry time stays a datetime."""


class QueueCreate(BaseModel, frozen=True):
    """A review queue's settings, as its creator gives them."""

    # frozen is a class keyword, not in model_config, so pyright sees the fields are
    # read-only and lets Queue narrow id; each subclass repeats it, as pyright requires.
    model_config = ConfigDict(extra='forbid')

    id: Name | None = None
    """Optional client id, for safe retries: creating the same id with the same settings
    again returns the stored queue, and other settings are refused. Generated when None."""

    name: Name
    description: Text | None = None
    instructions: Text | None = None
    """What reviewers are asked to do, for the UI to show with every item."""

    label_names: tuple[Name, ...] = ()
    """Names every review must give a label on the item's target, e.g. ``('passed', 'c1')``
    for a verdict and a rubric check. A review may give other labels too."""

    overlap: StrictInt = Field(default=1, ge=1, le=SQLITE_MAX_INT)
    """How many different reviewers review each item."""

    blind: StrictBool = True
    """Hide the labels on an item's kind and id from a reviewer until they submit their
    review of it (see :class:`ReviewView`)."""

    reservation_seconds: StrictInt = Field(default=3600, ge=1, le=MAX_RESERVATION_SECONDS)
    """How long a reviewer holds an item they were handed before it is free again."""

    reviewers: tuple[Name, ...] = ()
    """Who may ask for, view and review items; empty lets anyone with console access."""

    @model_validator(mode='after')
    def check_lists(self) -> QueueCreate:
        """Refuse repeated names, and an overlap more reviewers than the queue allows."""
        _check_unique('label_names', self.label_names)
        _check_unique('reviewers', self.reviewers)
        if self.reviewers and self.overlap > len(self.reviewers):
            raise ValueError(
                f'overlap {self.overlap} needs more reviewers than the {len(self.reviewers)} '
                'listed: no item could be fully reviewed'
            )
        return self


class Queue(QueueCreate, frozen=True):
    """A stored review queue."""

    # Required: Field() with no default tells pyright the parent's None isn't inherited.
    id: Name = Field()
    closed: bool = False
    """A closed queue hands out no items until it is reopened. A reviewer still holding a
    live reservation may submit it."""

    created_at: AwareDatetime


class QueueUpdate(BaseModel):
    """Changes to a queue; a field left out keeps its value. ``overlap``, ``blind`` and
    ``label_names`` never change."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    closed: StrictBool | None = None
    description: Text | None = None
    instructions: Text | None = None
    reservation_seconds: StrictInt | None = Field(default=None, ge=1, le=MAX_RESERVATION_SECONDS)
    """For reservations made or extended from then on."""

    reviewers: tuple[Name, ...] | None = None
    """The whole new list: every current reviewer, plus those added. A queue that lets
    anyone review keeps doing so."""

    @model_validator(mode='after')
    def check_given(self) -> QueueUpdate:
        """Refuse null where a value is needed, and repeated reviewers."""
        needed = ('closed', 'reservation_seconds', 'reviewers')
        nulls = [
            name for name in needed if name in self.model_fields_set and getattr(self, name) is None
        ]
        if nulls:
            raise ValueError(f'{nulls} cannot be null; leave a field out to keep its value')
        _check_unique('reviewers', self.reviewers or ())
        return self


class ItemCreate(BaseModel):
    """An item to review: what it is, and optional context such as the task text."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    target: LabelTarget
    context: Text | None = None


class QueueItem(ItemCreate):
    """A stored queue item."""

    id: Name
    queue_id: Name
    position: int
    """Items are handed out in position order: the order they were added in, from 1."""

    created_at: AwareDatetime


class Assignment(BaseModel):
    """One reviewer's turn at one item: a reservation and how it ended."""

    model_config = ConfigDict(frozen=True)

    id: str
    """Also the ``group_id`` of the labels the review stored."""

    queue_id: str
    item_id: str
    reviewer: str
    status: ReviewStatus
    """``reserved`` while held; then ``submitted``, ``released`` (handed back; it may be
    handed to the same reviewer again, after the items they haven't released), ``skipped``
    (never handed to this reviewer again) or ``expired`` (its time ran out). A released or
    expired turn can still be submitted while the item has a free slot."""

    reserved_at: AwareDatetime
    expires_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    """When it was submitted, released or skipped, or ran out; None while reserved."""


class ReviewLabel(BaseModel):
    """One label of a review. The queue sets the rest: a ``human`` source, the reviewer as
    author, ``blind`` from the queue and the reservation's id as ``group_id``."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    target: LabelTarget | None = None
    """The item's target or one inside it: the same kind and id, the same step and field
    where the item has one, and a span inside the item's. None is the item's target."""

    id: Name | None = None
    name: Name
    value: LabelValue
    comment: Text | None = None
    rubric_version: Name | None = None
    rubric_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')


class ReviewView(BaseModel):
    """An item as one reviewer may see it."""

    item: QueueItem
    assignment: Assignment | None = None
    """The reviewer's latest turn at the item; None if they never had one."""

    labels: list[Label]
    """Current labels on the item's kind and id, every step and span included, in ``seq``
    order; none until the reviewer has had a turn at the item. In a blind queue, none until
    they have submitted their review. Then only their review's own while they may still
    review another item on the same recording in a blind queue (one in a queue that is open,
    or that they hold a live reservation of, that lets them review, still needs reviews, and
    that they haven't reviewed or skipped); every one after that."""


class ItemProgress(BaseModel):
    """One item with its review counts."""

    item: QueueItem
    reviews: int
    """Reviews submitted."""

    needed: int
    """Reviews the item needs: the queue's overlap."""

    reserved: int
    """Reservations held now."""


class ItemPage(BaseModel):
    """One page of a queue's items, in position order."""

    items: list[ItemProgress]
    total: int
    """The queue's items across every page."""


class ReviewerProgress(BaseModel):
    """One reviewer's turns in a queue, by how they ended."""

    reviewer: str
    submitted: int
    reserved: int
    skipped: int
    released: int
    expired: int


class QueueProgress(BaseModel):
    """How far a queue's reviews are."""

    queue_id: str
    items: int
    complete: int
    """Items with every review they need."""

    in_progress: int
    """Items with a review or a reservation, but not every review they need."""

    reviews: int
    """Reviews submitted."""

    reviews_needed: int
    """Reviews the queue needs in all: items times overlap."""

    reserved: int
    """Reservations held now."""

    blocked: int
    """Items that can no longer get every review they need: the queue lists its reviewers
    and too many of them skipped the item. Adding reviewers unblocks them."""

    reviewers: list[ReviewerProgress]
    """Everyone who has had an item, by name."""


def _check_unique(field: str, values: Sequence[str]) -> None:
    """Refuse *values* that list a name twice."""
    if len(set(values)) != len(values):
        raise ValueError(f'{field} lists a name twice')
