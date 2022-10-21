from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .utils import distance, bearing, move


# ---------
# Geometry
# ---------
@dataclass
class Point2D(object):
    lng: float = 0.0
    lat: float = 0.0

    def array(self) -> List[float, float]:
        return [self.lng, self.lat]

    def tuple(self) -> Tuple[float, float]:
        return self.lng, self.lat

    def distance_to(self, other: Point2D):
        return distance(self.tuple(), other.tuple())

    def bearing_to(self, other: Point2D):
        return bearing(self.tuple(), other.tuple())

    def move_to(self, course: float, dist: float):
        self.lng, self.lat = move(self.tuple(), course, dist)

    def copy(self):
        return Point2D(self.lng, self.lat)

    def set(self, other: Point2D):
        self.lng = other.lng
        self.lat = other.lat

    def __str__(self):
        return '<%.5f,%.5f>' % (self.lng, self.lat)

    def __repr__(self):
        return '<%.5f,%.5f>' % (self.lng, self.lat)


@dataclass
class Waypoint:
    id: str
    location: Point2D

    def distance_to(self, other: Waypoint):
        return self.location.distance_to(other.location)

    def bearing_to(self, other: Waypoint):
        return self.location.bearing_to(other.location)

    def copy(self, name='Dogleg'):
        return Waypoint(id=name, location=self.location.copy())

    def __str__(self):
        return '[%s, %s]' % (self.id, self.location)

    def __repr__(self):
        return '[%s, %s]' % (self.id, self.location)


# ---------
# Operation
# ---------
@dataclass
class AircraftType:
    id: str
    normAcceleration: float
    maxAcceleration: float
    normDeceleration: float
    maxDeceleration: float
    liftOffSpeed: float
    flightPerformanceTable: List[Performance]

    def compute_performance(self, alt: float, val: Performance):
        if val.altitude == alt:
            return
        table = self.flightPerformanceTable
        if alt > len(table) * 100.0 - 100.0:
            val.copy(table[len(table) - 1])
            val.altitude = alt
            return
        if alt < table[0].altitude:
            val.copy(table[0])
            val.altitude = alt
            return
        idx = int(alt / 100.0)
        if alt % 100.0 == 0:
            val.copy(table[idx])
        else:
            f1 = table[idx]
            f2 = table[idx + 1]
            r = (f2.altitude - alt) / (f2.altitude - f1.altitude)
            f1.interpolate(r, f2, val)


@dataclass
class Aircraft:
    id: str
    aircraftType: AircraftType
    airline: str = None


@dataclass
class Routing:
    id: str
    wpt_list: List[Waypoint]

    def copy(self, section=None):
        if section is None:
            return Routing(self.id, self.wpt_list)
        return Routing(self.id, [self.wpt_list[i] for i in section])


@dataclass
class FlightPlan:
    id: str
    routing: Routing
    startTime: int
    aircraft: Aircraft
    min_alt: float
    max_alt: float

    def to_dict(self):
        return dict(id=self.id,
                    startTime=self.startTime,
                    min_alt=self.min_alt,
                    max_alt=self.max_alt,
                    register=self.aircraft.id,
                    type=self.aircraft.aircraftType.id,
                    route=self.routing.id,
                    wpt_list=[wpt.id for wpt in self.routing.wpt_list])


@dataclass
class DataSet:
    wpt_dict: Dict[str, Waypoint]
    rou_dict: Dict[str, Routing]
    fpl_dict: Dict[str, FlightPlan]
    air_dict: Dict[str, Aircraft]
    act_dict: Dict[str, AircraftType]


# ---------
# Conflict
# ---------
@dataclass
class Conflict:
    id: str
    time: int
    hDist: float
    vDist: float
    fpl0: FlightPlan
    fpl1: FlightPlan
    pos0: tuple
    pos1: tuple

    def __str__(self):
        return '%12s, %5d, %0.1f, %0.1f' % (self.id, self.time, self.hDist, self.vDist)

    def to_dict(self):
        return dict(id=self.id,
                    time=self.time,
                    h_dist=self.hDist,
                    v_dist=self.vDist)

    def printf(self):
        fpl0, fpl1 = self.fpl0, self.fpl1
        print('-------------------------------------')
        print('|  Conflict ID: ', self.id)
        print('|Conflict Time: ', self.time)
        print('|   H Distance: ', self.hDist)
        print('|   V Distance: ', self.vDist)
        print('|     a0 state: ', self.pos0)
        print('|      a0 info: ', fpl0.startTime, fpl0.min_alt, fpl0.max_alt, fpl0.routing.id)
        print('|     a1 state: ', self.pos1)
        print('|      a1 info: ', fpl1.startTime, fpl1.min_alt, fpl1.max_alt, fpl1.routing.id)
        print('-------------------------------------')


# ---------
# Performance
# ---------
@dataclass
class Performance:
    altitude: float = 0
    minClimbTAS: float = 0
    normClimbTAS: float = 0
    maxClimbTAS: float = 0
    climbFuel: float = 0
    minDescentTAS: float = 0
    normDescentTAS: float = 0
    maxDescentTAS: float = 0
    descentFuel: float = 0
    minCruiseTAS: float = 0
    normCruiseTAS: float = 0
    maxCruiseTAS: float = 0
    cruiseFuel: float = 0
    normClimbRate: float = 0
    maxClimbRate: float = 0
    normDescentRate: float = 0
    maxDescentRate: float = 0
    normTurnRate: float = 0
    maxTurnRate: float = 0

    def interpolate(self, r: float, other: Performance, val: Performance):
        # pylint: disable=no-member
        for k in self.__dataclass_fields__:
            v1 = getattr(self, k)
            v2 = getattr(other, k)
            v = (v2 - v1) * r + v1
            setattr(val, k, v)

    def copy(self, other):
        # pylint: disable=no-member
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(other, k))

    def set(self, other):
        # pylint: disable=no-member
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(other, k))
