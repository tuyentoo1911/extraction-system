import React from 'react';
import { User, Building2, MapPin, Package, Calendar, DollarSign, Clock, Factory, Percent, HelpCircle } from 'lucide-react';

export const ICON_PATHS: Record<string, Path2D[]> = {
  Person: [
    new Path2D("M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"),
    new Path2D("M12 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8z")
  ],
  Organization: [
    new Path2D("M6 22V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v18Z"),
    new Path2D("M6 12H4a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h2"),
    new Path2D("M18 9h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-2"),
    new Path2D("M10 6h4"),
    new Path2D("M10 10h4"),
    new Path2D("M10 14h4"),
    new Path2D("M10 18h4")
  ],
  Location: [
    new Path2D("M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"),
    new Path2D("M12 13a3 3 0 1 0 0-6 3 3 0 0 0 0 6z")
  ],
  Product: [
    new Path2D("m7.5 4.27 9 5.15"),
    new Path2D("M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"),
    new Path2D("m3.3 7 8.7 5 8.7-5"),
    new Path2D("M12 22V12")
  ],
  Event: [
    new Path2D("M8 2v4"),
    new Path2D("M16 2v4"),
    new Path2D("M5 4h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z"),
    new Path2D("M3 10h18")
  ],
  Money: [
    new Path2D("M12 2v20"),
    new Path2D("M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6")
  ],
  Date: [
    new Path2D("M8 2v4"),
    new Path2D("M16 2v4"),
    new Path2D("M3 10h18"),
    new Path2D("M5 4h14a2 2 0 0 1 2 2v5H3V6a2 2 0 0 1 2-2z"),
    new Path2D("M3 10v10a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V10")
  ],
  Industry: [
    new Path2D("M2 20a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V8l-7-6H4a2 2 0 0 0-2 2v16Z"),
    new Path2D("M15 2v6h6"),
    new Path2D("M7 13h4"),
    new Path2D("M7 17h4")
  ],
  Percent: [
    new Path2D("M19 5 5 19"),
    new Path2D("M6.5 6.5a1 1 0 1 0 2 0 1 1 0 0 0-2 0"),
    new Path2D("M16.5 16.5a1 1 0 1 0 2 0 1 1 0 0 0-2 0")
  ],
  Default: [
    new Path2D("M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"),
    new Path2D("M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"),
    new Path2D("M12 17h.01")
  ]
};

export const TYPE_COLORS: Record<string, string> = {
  Person:       '#2563eb',   // blue
  Organization: '#9333ea',   // purple
  Location:     '#16a34a',   // green
  Product:      '#ea580c',   // orange
  Event:        '#dc2626',   // red
  Money:        '#ca8a04',   // yellow/gold
  Date:         '#0891b2',   // cyan
  Industry:     '#7c3aed',   // violet
  Percent:      '#db2777',   // pink
};

export const TYPE_BADGE_COLORS: Record<string, string> = {
  Person:       'bg-blue-100 text-blue-800 border-blue-300',
  Organization: 'bg-purple-100 text-purple-800 border-purple-300',
  Location:     'bg-green-100 text-green-800 border-green-300',
  Product:      'bg-orange-100 text-orange-800 border-orange-300',
  Event:        'bg-red-100 text-red-800 border-red-300',
  Money:        'bg-yellow-100 text-yellow-800 border-yellow-300',
  Date:         'bg-cyan-100 text-cyan-800 border-cyan-300',
  Industry:     'bg-violet-100 text-violet-800 border-violet-300',
  Percent:      'bg-pink-100 text-pink-800 border-pink-300',
};

export function getTypeIcon(type: string): React.ReactNode {
  switch (type.toLowerCase()) {
    case 'person':       return <User className="w-3 h-3" />;
    case 'organization': return <Building2 className="w-3 h-3" />;
    case 'location':     return <MapPin className="w-3 h-3" />;
    case 'product':      return <Package className="w-3 h-3" />;
    case 'event':        return <Calendar className="w-3 h-3" />;
    case 'money':        return <DollarSign className="w-3 h-3" />;
    case 'date':         return <Clock className="w-3 h-3" />;
    case 'industry':     return <Factory className="w-3 h-3" />;
    case 'percent':      return <Percent className="w-3 h-3" />;
    default:             return <HelpCircle className="w-3 h-3" />;
  }
}
