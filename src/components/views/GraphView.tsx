import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { Orbit, Plus, Minus, Maximize, Eye, Edit2, Trash2, Zap, Database } from 'lucide-react';
import ForceGraph2D from 'react-force-graph-2d';
import * as d3 from 'd3';
import { ICON_PATHS, TYPE_COLORS, getTypeIcon } from '../../constants/graph';
import type { GraphData } from '../../types';

interface GraphViewProps {
  data: GraphData;
}

export default function GraphView({ data }: GraphViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<any>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [hoverNode, setHoverNode] = useState<any>(null);
  const [hoverLink, setHoverLink] = useState<any>(null);
  const [selectedNode, setSelectedNode] = useState<any>(null);
  const [contextMenu, setContextMenu] = useState<{ show: boolean; x: number; y: number; type: 'node' | 'link'; data: any } | null>(null);
  const [layoutMode, setLayoutMode] = useState<'force' | 'td' | 'lr' | 'radialout'>('force');
  const [physicsEnabled, setPhysicsEnabled] = useState(true);

  const lastHoverNode = useRef<any>(null);
  if (hoverNode) lastHoverNode.current = hoverNode;
  const displayNode = hoverNode || lastHoverNode.current;

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (tooltipRef.current) {
        tooltipRef.current.style.left = `${e.clientX}px`;
        tooltipRef.current.style.top = `${e.clientY}px`;
      }
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver(entries => {
      if (entries[0]) {
        const { width, height } = entries[0].contentRect;
        setDimensions({ width, height });
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  const graphDataMemo = useMemo(() => {
    const nodes = data.entities.map(e => ({ ...e, degree: 0, isCentral: false }));
    const links = data.relations.map(r => ({ ...r }));

    const linkCounts: Record<string, number> = {};
    const degrees: Record<string, number> = {};

    links.forEach((l: any) => {
      const s = typeof l.source === 'object' ? l.source.id : l.source;
      const t = typeof l.target === 'object' ? l.target.id : l.target;
      degrees[s] = (degrees[s] || 0) + 1;
      degrees[t] = (degrees[t] || 0) + 1;
      const key = s < t ? `${s}-${t}` : `${t}-${s}`;
      linkCounts[key] = (linkCounts[key] || 0) + 1;
      l.linkIndex = linkCounts[key];
    });

    let maxDegree = 0;
    let centralNodeId: string | null = null;
    Object.entries(degrees).forEach(([id, degree]) => {
      if (degree > maxDegree) { maxDegree = degree; centralNodeId = id; }
    });

    nodes.forEach((n: any) => {
      n.degree = degrees[n.id] || 0;
      n.isCentral = n.id === centralNodeId;
    });

    links.forEach((l: any) => {
      const s = typeof l.source === 'object' ? l.source.id : l.source;
      const t = typeof l.target === 'object' ? l.target.id : l.target;
      const key = s < t ? `${s}-${t}` : `${t}-${s}`;
      l.totalLinks = linkCounts[key];
      if (s === t) {
        l.curvature = 0.5;
      } else if (l.totalLinks > 1) {
        const offset = l.linkIndex - 1 - (l.totalLinks - 1) / 2;
        l.curvature = (s < t ? 1 : -1) * offset * 0.4;
      } else {
        l.curvature = 0;
      }
    });

    return { nodes, links };
  }, [data]);

  const { nodes, links } = graphDataMemo;

  const handleZoomIn = useCallback(() => {
    if (fgRef.current) fgRef.current.zoom(fgRef.current.zoom() * 1.5, 400);
  }, []);

  const handleZoomOut = useCallback(() => {
    if (fgRef.current) fgRef.current.zoom(fgRef.current.zoom() / 1.5, 400);
  }, []);

  const handleZoomFit = useCallback(() => {
    if (fgRef.current) fgRef.current.zoomToFit(400, 50);
  }, []);

  const closeContextMenu = useCallback(() => setContextMenu(null), []);

  const handleNodeRightClick = useCallback((node: any, event: MouseEvent) => {
    event.preventDefault();
    setContextMenu({ show: true, x: event.clientX, y: event.clientY, type: 'node', data: node });
  }, []);

  const handleLinkRightClick = useCallback((link: any, event: MouseEvent) => {
    event.preventDefault();
    setContextMenu({ show: true, x: event.clientX, y: event.clientY, type: 'link', data: link });
  }, []);

  useEffect(() => {
    if (!fgRef.current) return;
    const isDense = nodes.length > 20;
    const usePhysics = physicsEnabled && layoutMode === 'force';

    nodes.forEach((n: any) => {
      if (usePhysics) {
        n.fx = undefined;
        n.fy = undefined;
      } else {
        n.fx = Number.isFinite(n.x) ? n.x : 0;
        n.fy = Number.isFinite(n.y) ? n.y : 0;
      }
    });

    fgRef.current.d3Force('collide', d3.forceCollide().radius((node: any) => node.isCentral ? 42 : (isDense ? 24 : 30)).iterations(2));
    fgRef.current.d3Force('charge').strength(usePhysics ? (isDense ? -560 : -460) : 0).distanceMax(700);
    fgRef.current.d3Force('link').distance(isDense ? 75 : 100);
    fgRef.current.d3Force('x', d3.forceX().strength(usePhysics ? (isDense ? 0.1 : 0.08) : 0));
    fgRef.current.d3Force('y', d3.forceY().strength(usePhysics ? (isDense ? 0.1 : 0.08) : 0));
    fgRef.current.d3ReheatSimulation();
  }, [nodes, links, layoutMode, physicsEnabled]);

  const highlightNodes = useMemo(() => {
    const set = new Set<string>();
    if (selectedNode) {
      set.add(selectedNode.id);
      links.forEach((l: any) => {
        const s = typeof l.source === 'object' ? l.source.id : l.source;
        const t = typeof l.target === 'object' ? l.target.id : l.target;
        if (s === selectedNode.id) set.add(t);
        if (t === selectedNode.id) set.add(s);
      });
    }
    return set;
  }, [selectedNode, links]);

  const highlightLinks = useMemo(() => {
    const set = new Set<any>();
    if (hoverLink) set.add(hoverLink);
    if (selectedNode) {
      links.forEach((l: any) => {
        const s = typeof l.source === 'object' ? l.source.id : l.source;
        const t = typeof l.target === 'object' ? l.target.id : l.target;
        if (s === selectedNode.id || t === selectedNode.id) set.add(l);
      });
    }
    return set;
  }, [selectedNode, hoverLink, links]);

  const drawNode = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const isHighlighted = selectedNode ? highlightNodes.has(node.id) : true;
    const isHovered = hoverNode?.id === node.id;
    ctx.globalAlpha = isHighlighted ? 1 : 0.15;

    const isLargeGraph = nodes.length > 15;
    const threshold = isLargeGraph ? 1.2 : 0.6;
    const scaleMultiplier = node.isCentral ? 1.5 : 1;

    if (globalScale < threshold && !isHovered && !isHighlighted) {
      ctx.beginPath();
      ctx.arc(node.x, node.y, 4 * scaleMultiplier, 0, 2 * Math.PI, false);
      ctx.fillStyle = TYPE_COLORS[node.type] || '#999';
      ctx.fill();
      ctx.lineWidth = 0.5 * scaleMultiplier;
      ctx.strokeStyle = '#000';
      ctx.stroke();
      ctx.globalAlpha = 1;
      return;
    }

    const fontSize = 5 * scaleMultiplier;
    const typeFontSize = 3 * scaleMultiplier;
    const iconSize = 5 * scaleMultiplier;
    const iconPadding = 2 * scaleMultiplier;
    const paddingX = 4 * scaleMultiplier;
    const paddingY = 3 * scaleMultiplier;
    const colorBarWidth = 3 * scaleMultiplier;

    ctx.font = `bold ${fontSize}px Inter, sans-serif`;
    const textWidth = ctx.measureText(node.name).width;
    ctx.font = `${typeFontSize}px JetBrains Mono, monospace`;
    const typeWidth = ctx.measureText(node.type.toUpperCase()).width;

    const contentWidth = Math.max(iconSize + iconPadding + textWidth, typeWidth);
    const width = contentWidth + paddingX * 2 + colorBarWidth;
    const height = fontSize + typeFontSize + paddingY * 3;
    const x = node.x - width / 2;
    const y = node.y - height / 2;

    if (isHovered || (selectedNode && isHighlighted)) {
      ctx.fillStyle = 'rgba(0,0,0,1)';
      ctx.fillRect(x + 1.5 * scaleMultiplier, y + 1.5 * scaleMultiplier, width, height);
    }

    ctx.fillStyle = '#fff';
    ctx.fillRect(x, y, width, height);
    ctx.lineWidth = 0.5 * scaleMultiplier;
    ctx.strokeStyle = '#000';
    ctx.strokeRect(x, y, width, height);
    ctx.fillStyle = TYPE_COLORS[node.type] || '#999';
    ctx.fillRect(x, y, colorBarWidth, height);

    ctx.save();
    ctx.translate(x + paddingX + colorBarWidth, y + paddingY);
    ctx.scale(iconSize / 24, iconSize / 24);
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    (ICON_PATHS[node.type] || ICON_PATHS.Default).forEach(p => ctx.stroke(p));
    ctx.restore();

    ctx.fillStyle = '#000';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.font = `bold ${fontSize}px Inter, sans-serif`;
    ctx.fillText(node.name, x + paddingX + colorBarWidth + iconSize + iconPadding, y + paddingY);
    ctx.fillStyle = '#666';
    ctx.font = `${typeFontSize}px JetBrains Mono, monospace`;
    ctx.fillText(node.type.toUpperCase(), x + paddingX + colorBarWidth, y + paddingY * 2 + fontSize);
    ctx.globalAlpha = 1;
  }, [selectedNode, highlightNodes, hoverNode, nodes.length]);

  const drawLink = useCallback((link: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const isHighlighted = selectedNode ? highlightLinks.has(link) : true;
    const isHovered = hoverLink === link;
    if (!isHighlighted && !isHovered) return;
    if (!(globalScale >= 1.5 || (selectedNode && isHighlighted) || isHovered)) return;

    const start = link.source;
    const end = link.target;
    if (typeof start !== 'object' || typeof end !== 'object') return;

    let pos: { x: number; y: number };
    if (start === end || (start.x === end.x && start.y === end.y)) {
      pos = { x: start.x, y: start.y - 12 };
    } else if (link.curvature === 0) {
      pos = { x: start.x + (end.x - start.x) / 2, y: start.y + (end.y - start.y) / 2 };
    } else {
      const dx = end.x - start.x;
      const dy = end.y - start.y;
      const l = Math.sqrt(dx * dx + dy * dy);
      const offset = l * link.curvature * 0.5;
      pos = {
        x: start.x + dx / 2 + (-dy / l) * offset,
        y: start.y + dy / 2 + (dx / l) * offset,
      };
    }

    const label = link.label.toUpperCase();
    const fontSize = 3;
    ctx.font = `${fontSize}px JetBrains Mono, monospace`;
    const textWidth = ctx.measureText(label).width;
    const pX = 2, pY = 1.5;
    const bgW = textWidth + pX * 2;
    const bgH = fontSize + pY * 2;
    const bx = pos.x - bgW / 2;
    const by = pos.y - bgH / 2;

    ctx.fillStyle = link.isPredicted ? '#ecfdf5' : '#f4f4f0';
    ctx.fillRect(bx, by, bgW, bgH);
    ctx.lineWidth = 0.2;
    ctx.strokeStyle = link.isPredicted ? '#10b981' : '#000';
    ctx.strokeRect(bx, by, bgW, bgH);
    ctx.fillStyle = link.isPredicted ? '#10b981' : '#f25f22';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, pos.x, pos.y + 0.3);
  }, [selectedNode, highlightLinks, hoverLink]);

  return (
    <div ref={containerRef} className="w-full h-full bg-white relative overflow-hidden flex items-center justify-center">
      <div className="absolute top-4 left-4 font-mono text-[10px] text-black/40 uppercase tracking-widest pointer-events-none z-10">
        Cuộn để thu phóng / Kéo để di chuyển / Click để làm nổi bật
      </div>

      {dimensions.width > 0 && (
        <ForceGraph2D
          ref={fgRef}
          width={dimensions.width}
          height={dimensions.height}
          graphData={graphDataMemo}
          dagMode={layoutMode === 'force' ? undefined : layoutMode}
          dagLevelDistance={80}
          nodeLabel=""
          nodeVal={25}
          nodeCanvasObject={drawNode}
          linkCanvasObjectMode={() => 'after'}
          linkCanvasObject={drawLink}
          linkColor={(link: any) => {
            if (hoverLink === link) return '#f25f22';
            const hi = selectedNode ? highlightLinks.has(link) : true;
            if (!hi) return 'rgba(0,0,0,0.1)';
            return link.isPredicted ? '#10b981' : (selectedNode ? '#f25f22' : '#000');
          }}
          linkWidth={(link: any) => {
            if (hoverLink === link) return 2;
            const hi = selectedNode ? highlightLinks.has(link) : true;
            return hi ? (link.isPredicted ? 1.5 : 1) : 0.2;
          }}
          linkLineDash={(link: any) => link.isPredicted ? [4, 4] : undefined}
          linkCurvature="curvature"
          linkDirectionalArrowLength={(link: any) => (selectedNode ? highlightLinks.has(link) : true) ? 3 : 0}
          linkDirectionalArrowRelPos={1}
          linkDirectionalArrowColor={(link: any) => {
            if (hoverLink === link) return '#f25f22';
            const hi = selectedNode ? highlightLinks.has(link) : true;
            if (!hi) return 'rgba(0,0,0,0.1)';
            return link.isPredicted ? '#10b981' : (selectedNode ? '#f25f22' : '#000');
          }}
          linkDirectionalParticles={(link: any) => (hoverLink === link || (selectedNode && highlightLinks.has(link))) ? 2 : 0}
          linkDirectionalParticleWidth={2}
          linkDirectionalParticleSpeed={0.01}
          linkDirectionalParticleColor={() => '#f25f22'}
          onNodeHover={setHoverNode}
          onLinkHover={setHoverLink}
          onNodeClick={(node) => { setSelectedNode(node.id === selectedNode?.id ? null : node); closeContextMenu(); }}
          onNodeRightClick={handleNodeRightClick}
          onLinkRightClick={handleLinkRightClick}
          onBackgroundClick={() => { setSelectedNode(null); closeContextMenu(); }}
          onBackgroundRightClick={closeContextMenu}
          d3AlphaDecay={0.02}
          d3VelocityDecay={0.3}
          cooldownTicks={100}
        />
      )}

      <div className="absolute top-4 right-4 flex items-center gap-1 z-10 bg-white/80 backdrop-blur-sm p-1 border border-black/10 shadow-sm">
        <button
          onClick={() => setPhysicsEnabled((v) => !v)}
          className={`w-8 h-8 flex items-center justify-center transition-colors border ${
            physicsEnabled
              ? 'bg-black text-white border-black'
              : 'bg-white text-black/60 border-black/20 hover:bg-[#f25f22] hover:border-[#f25f22] hover:text-white active:bg-black'
          }`}
          title={physicsEnabled ? 'Tắt mô phỏng lực' : 'Bật mô phỏng lực'}
        >
          <Zap className="w-4 h-4" />
        </button>
        {([
          { mode: 'force', title: 'Bố cục tự do', icon: <Orbit className="w-4 h-4" /> },
          { mode: 'td', title: 'Bố cục cây dọc', glyph: '↓' },
          { mode: 'lr', title: 'Bố cục cây ngang', glyph: '→' },
          { mode: 'radialout', title: 'Bố cục tỏa tròn', glyph: '◎' },
        ] as const).map(({ mode, title, icon, glyph }) => (
          <button
            key={mode}
            onClick={() => setLayoutMode(mode)}
            className={`w-8 h-8 flex items-center justify-center transition-colors border ${
              layoutMode === mode
                ? 'bg-black text-white border-black'
                : 'bg-white text-black/60 border-black/20 hover:bg-[#f25f22] hover:border-[#f25f22] hover:text-white active:bg-black'
            }`}
            title={title}
          >
            {icon ?? <span className="font-mono text-sm font-bold leading-none">{glyph}</span>}
          </button>
        ))}
      </div>

      <div className="absolute bottom-4 left-4 flex flex-col gap-2 z-10">
        {[
          { onClick: handleZoomIn, title: 'Zoom In', icon: <Plus className="w-4 h-4" /> },
          { onClick: handleZoomOut, title: 'Zoom Out', icon: <Minus className="w-4 h-4" /> },
          { onClick: handleZoomFit, title: 'Fit to Screen', icon: <Maximize className="w-4 h-4" /> },
        ].map(({ onClick, title, icon }) => (
          <button
            key={title}
            onClick={onClick}
            title={title}
            className="w-8 h-8 bg-white border border-black flex items-center justify-center hover:bg-[#f25f22] hover:text-white active:bg-black transition-colors shadow-[2px_2px_0_rgba(0,0,0,1)]"
          >
            {icon}
          </button>
        ))}
      </div>

      {contextMenu?.show && (
        <div
          className="fixed z-50 bg-white border border-black shadow-[4px_4px_0_rgba(0,0,0,1)] flex flex-col min-w-[150px]"
          style={{ left: contextMenu.x, top: contextMenu.y }}
        >
          <div className="px-3 py-2 border-b border-black/10 bg-[#f4f4f0] font-mono text-[10px] uppercase tracking-widest text-black/50">
            {contextMenu.type === 'node' ? 'Node Actions' : 'Link Actions'}
          </div>
          <button className="px-4 py-2 text-left text-sm hover:bg-black hover:text-white transition-colors flex items-center gap-2"
            onClick={() => { alert(`View: ${contextMenu.data.name || contextMenu.data.label}`); closeContextMenu(); }}>
            <Eye className="w-4 h-4" /> View Details
          </button>
          <button className="px-4 py-2 text-left text-sm hover:bg-black hover:text-white transition-colors flex items-center gap-2"
            onClick={() => { alert(`Edit: ${contextMenu.data.name || contextMenu.data.label}`); closeContextMenu(); }}>
            <Edit2 className="w-4 h-4" /> Edit
          </button>
          <button className="px-4 py-2 text-left text-sm text-red-600 hover:bg-red-600 hover:text-white transition-colors flex items-center gap-2"
            onClick={() => { alert(`Delete: ${contextMenu.data.name || contextMenu.data.label}`); closeContextMenu(); }}>
            <Trash2 className="w-4 h-4" /> Delete
          </button>
        </div>
      )}

      <div
        ref={tooltipRef}
        className="fixed z-50 pointer-events-none bg-white text-black p-3 border border-black shadow-[4px_4px_0_rgba(0,0,0,1)] transition-opacity duration-150"
        style={{ opacity: hoverNode && !selectedNode ? 1 : 0, transform: 'translate(16px, 16px)', left: -9999, top: -9999 }}
      >
        {displayNode && (
          <>
            <div className="font-mono text-[10px] text-black/50 uppercase tracking-widest mb-2 flex items-center gap-2">
              <span className="flex items-center justify-center w-5 h-5 rounded-full bg-black/5">
                {getTypeIcon(displayNode.type)}
              </span>
              <span>{displayNode.id} • {displayNode.type}</span>
            </div>
            <div className="font-bold text-sm mb-1">{displayNode.name}</div>
            <div className="font-mono text-[9px] text-black/40 uppercase">Click để làm nổi bật</div>
          </>
        )}
      </div>

      {selectedNode && (
        <div className="absolute top-4 right-4 bottom-4 w-96 bg-white border border-black shadow-[4px_4px_0_rgba(0,0,0,1)] flex flex-col z-20 overflow-hidden animate-in slide-in-from-right-8">
          <div className="p-4 border-b border-black/10 bg-[#f4f4f0] flex items-center justify-between flex-shrink-0">
            <h3 className="font-mono text-xs font-bold tracking-widest uppercase flex items-center gap-2">
              <Database className="w-4 h-4" /> Chi tiết thực thể
            </h3>
            <button onClick={() => setSelectedNode(null)} className="text-black/50 hover:text-black transition-colors">
              <Minus className="w-4 h-4" />
            </button>
          </div>
          <div className="p-6 overflow-y-auto flex-1">
            <div className="flex items-start gap-4 mb-8">
              <div className="w-12 h-12 rounded-full bg-[#f4f4f0] border border-black/10 flex items-center justify-center flex-shrink-0 mt-1">
                {getTypeIcon(selectedNode.type)}
              </div>
              <div className="flex-1 min-w-0">
                <div className="font-mono text-[10px] text-black/50 uppercase tracking-widest mb-1.5">{selectedNode.id}</div>
                <h2 className="text-xl font-bold leading-snug break-words">{selectedNode.name}</h2>
              </div>
            </div>
            <table className="w-full text-sm text-left border-collapse">
              <thead>
                <tr className="border-b border-black/20">
                  <th className="pb-3 font-mono text-[10px] uppercase tracking-widest text-black/50 font-normal w-1/3">Thuộc tính</th>
                  <th className="pb-3 font-mono text-[10px] uppercase tracking-widest text-black/50 font-normal">Giá trị</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-black/5">
                  <td className="py-4 font-mono text-xs text-black/70 align-top pr-4">Type</td>
                  <td className="py-4 font-bold break-words">{selectedNode.type}</td>
                </tr>
                {selectedNode.properties?.map((prop: any, idx: number) => (
                  <tr key={idx} className="border-b border-black/5 last:border-0">
                    <td className="py-4 font-mono text-xs text-black/70 align-top pr-4">{prop.key}</td>
                    <td className="py-4 font-bold break-words leading-relaxed">{prop.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
