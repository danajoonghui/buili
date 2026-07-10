"use client";

import { AlertTriangle, Box, Check, Loader2, Maximize, RefreshCcw } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { api } from "@/lib/api";

export type SpatialViewerPin = {
  issueId: string;
  code: string;
  title: string;
  x: number;
  y: number;
  tone: string;
};

type Props = {
  projectId: string;
  initialAssetId?: string;
  pins: SpatialViewerPin[];
  selectedIssueId?: string;
  geometryConfidence?: number;
  onSelectIssue: (issueId: string) => void;
  fallbackImage: string;
  compact?: boolean;
};

type ViewerState = "preparing" | "loading" | "ready" | "fallback";

export function SpatialModelViewer({ projectId, initialAssetId, pins, selectedIssueId, geometryConfidence = 0, onSelectIssue, fallbackImage, compact = false }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const resetCameraRef = useRef<(() => void) | null>(null);
  const issueMeshesRef = useRef<THREE.Mesh[]>([]);
  const selectedRef = useRef(selectedIssueId);
  const selectRef = useRef(onSelectIssue);
  const [assetId,setAssetId]=useState(initialAssetId??"");
  const [state,setState]=useState<ViewerState>(initialAssetId?"loading":"preparing");
  const [message,setMessage]=useState(initialAssetId?"Loading generated geometry…":"Preparing the 3D spatial index…");
  const [attempt,setAttempt]=useState(0);

  useEffect(()=>{selectedRef.current=selectedIssueId},[selectedIssueId]);
  useEffect(()=>{selectRef.current=onSelectIssue},[onSelectIssue]);
  useEffect(()=>{
    issueMeshesRef.current.forEach(mesh=>{
      const active=mesh.userData.issueId===selectedIssueId;
      const material=mesh.material;
      const materials=Array.isArray(material)?material:[material];
      materials.forEach(item=>{
        if(item instanceof THREE.MeshStandardMaterial){
          item.color.set(active?0xd37a26:0x174d3b);
          item.emissive.set(active?0x5b2105:0x061c14);
          item.emissiveIntensity=active?.42:.18;
        }
      });
      mesh.scale.setScalar(active?1.32:1);
    });
  },[selectedIssueId]);

  const retry=useCallback(()=>{
    setAssetId(initialAssetId??"");
    setMessage(initialAssetId?"Loading generated geometry…":"Preparing the 3D spatial index…");
    setState(initialAssetId?"loading":"preparing");
    setAttempt(value=>value+1);
  },[initialAssetId]);

  useEffect(()=>{
    if(assetId||!projectId) return;
    let active=true;
    api.createDesign3d(projectId).then(asset=>{
      if(!active)return;
      setAssetId(asset.id);
      setState("loading");
      setMessage("Loading generated geometry…");
    }).catch(()=>{
      if(!active)return;
      setState("fallback");
      setMessage("Interactive geometry is not available yet. Run verification to rebuild it.");
    });
    return()=>{active=false};
  },[assetId,attempt,projectId]);

  useEffect(()=>{
    const host=hostRef.current;
    if(!host||!assetId)return;
    const container=host;
    let renderer:THREE.WebGLRenderer;
    try{
      renderer=new THREE.WebGLRenderer({antialias:true,alpha:false,powerPreference:"high-performance"});
    }catch{
      setState("fallback");
      setMessage("WebGL is unavailable on this device. The synchronized 3D snapshot is shown instead.");
      return;
    }
    let disposed=false;
    let animation=0;
    const scene=new THREE.Scene();
    scene.background=new THREE.Color(0xf1f3f0);
    const camera=new THREE.PerspectiveCamera(42,1,.01,5000);
    const controls=new OrbitControls(camera,renderer.domElement);
    controls.enableDamping=true;
    controls.dampingFactor=.08;
    controls.screenSpacePanning=true;
    controls.minDistance=.5;
    controls.maxDistance=250;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
    renderer.outputColorSpace=THREE.SRGBColorSpace;
    renderer.toneMapping=THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure=1.05;
    renderer.domElement.setAttribute("aria-label","Interactive generated 3D construction context. Drag to orbit, shift-drag to pan, and scroll to zoom.");
    renderer.domElement.setAttribute("role","img");
    renderer.domElement.tabIndex=0;
    container.replaceChildren(renderer.domElement);

    scene.add(new THREE.HemisphereLight(0xffffff,0x66756d,2.1));
    const key=new THREE.DirectionalLight(0xffffff,2.8);
    key.position.set(12,18,9);
    scene.add(key);
    const fill=new THREE.DirectionalLight(0xb7d7c6,1.1);
    fill.position.set(-8,7,-12);
    scene.add(fill);

    const raycaster=new THREE.Raycaster();
    const pointer=new THREE.Vector2();
    const issueMeshes:THREE.Mesh[]=[];
    let cameraHome={position:new THREE.Vector3(4,4,6),target:new THREE.Vector3()};

    function resetCamera(){
      camera.position.copy(cameraHome.position);
      controls.target.copy(cameraHome.target);
      controls.update();
    }
    resetCameraRef.current=resetCamera;

    function resize(){
      const width=Math.max(container.clientWidth,320);
      const height=Math.max(container.clientHeight,compact?300:430);
      camera.aspect=width/height;
      camera.updateProjectionMatrix();
      renderer.setSize(width,height,false);
    }
    resize();
    const resizeObserver=new ResizeObserver(resize);
    resizeObserver.observe(container);

    function chooseIssue(event:PointerEvent){
      const rect=renderer.domElement.getBoundingClientRect();
      pointer.x=((event.clientX-rect.left)/rect.width)*2-1;
      pointer.y=-((event.clientY-rect.top)/rect.height)*2+1;
      raycaster.setFromCamera(pointer,camera);
      const hit=raycaster.intersectObjects(issueMeshes,false)[0];
      const id=hit?.object.userData.issueId as string|undefined;
      if(id)selectRef.current(id);
    }
    renderer.domElement.addEventListener("pointerup",chooseIssue);

    const loader=new GLTFLoader();
    loader.load(`/api/v1/spatial-assets/${encodeURIComponent(assetId)}/download`,gltf=>{
      if(disposed)return;
      const model=gltf.scene;
      scene.add(model);
      model.traverse(object=>{
        if(!(object instanceof THREE.Mesh))return;
        object.castShadow=false;
        object.receiveShadow=true;
        const materials=Array.isArray(object.material)?object.material:[object.material];
        materials.forEach(material=>{
          if(material instanceof THREE.MeshStandardMaterial){
            material.roughness=.86;
            material.metalness=.02;
          }
        });
      });
      const bounds=new THREE.Box3().setFromObject(model);
      const size=bounds.getSize(new THREE.Vector3());
      const center=bounds.getCenter(new THREE.Vector3());
      const radius=Math.max(size.length()*.56,1);
      const floorY=bounds.max.y+Math.max(size.y*.06,.08);
      pins.forEach(pin=>{
        const active=pin.issueId===selectedRef.current;
        const material=new THREE.MeshStandardMaterial({color:active?0xd37a26:0x174d3b,emissive:active?0x5b2105:0x061c14,emissiveIntensity:active?.42:.18,roughness:.55});
        const marker=new THREE.Mesh(new THREE.SphereGeometry(Math.max(radius*.018,.035),18,12),material);
        marker.position.set(bounds.min.x+(pin.x/100)*size.x,floorY,bounds.min.z+(pin.y/100)*size.z);
        marker.userData.issueId=pin.issueId;
        marker.userData.label=`${pin.code} ${pin.title}`;
        issueMeshes.push(marker);
        scene.add(marker);
        const stem=new THREE.Mesh(new THREE.CylinderGeometry(Math.max(radius*.003,.006),Math.max(radius*.003,.006),Math.max(radius*.08,.12),10),material.clone());
        stem.position.copy(marker.position);
        stem.position.y-=Math.max(radius*.04,.06);
        stem.userData.issueId=pin.issueId;
        issueMeshes.push(stem);
        scene.add(stem);
      });
      issueMeshesRef.current=issueMeshes;
      const grid=new THREE.GridHelper(Math.max(size.x,size.z,radius)*1.35,20,0x9eb2a7,0xd9dfdb);
      grid.position.set(center.x,bounds.min.y-.01,center.z);
      scene.add(grid);
      cameraHome={position:new THREE.Vector3(center.x+radius*.7,center.y+radius*.62,center.z+radius*.78),target:center.clone()};
      resetCamera();
      setState("ready");
      setMessage("Interactive model ready");
    },event=>{
      if(event.total>0)setMessage(`Loading generated geometry · ${Math.round(event.loaded/event.total*100)}%`);
    },()=>{
      if(disposed)return;
      setState("fallback");
      setMessage("The generated GLB could not be loaded. The synchronized spatial snapshot is shown instead.");
    });

    function animate(){
      controls.update();
      renderer.render(scene,camera);
      animation=requestAnimationFrame(animate);
    }
    animate();
    return()=>{
      disposed=true;
      cancelAnimationFrame(animation);
      resizeObserver.disconnect();
      renderer.domElement.removeEventListener("pointerup",chooseIssue);
      controls.dispose();
      scene.traverse(object=>{
        if(object instanceof THREE.Mesh){
          object.geometry.dispose();
          const materials=Array.isArray(object.material)?object.material:[object.material];
          materials.forEach(material=>material.dispose());
        }
      });
      renderer.dispose();
      renderer.domElement.remove();
      issueMeshesRef.current=[];
      resetCameraRef.current=null;
    };
  },[assetId,attempt,compact,pins]);

  return <section className={`spatial-model-viewer ${compact?"compact":""}`} aria-label="Generated 3D model viewer">
    <div className="spatial-webgl-host" ref={hostRef}/>
    {state!=="ready"?<div className={`spatial-model-state ${state}`} aria-live="polite">{state==="fallback"?<img src={fallbackImage} alt="Synchronized 3D spatial snapshot fallback"/>:null}<span>{state==="fallback"?<AlertTriangle size={18}/>:<Loader2 className="spin" size={20}/>}<b>{message}</b>{state==="fallback"?<button type="button" onClick={retry}><RefreshCcw size={15}/> Retry model</button>:null}</span></div>:null}
    <div className="spatial-model-hud"><span><Box size={14}/> Generated from current 2D source</span><span className={geometryConfidence>=.8?"ready":"review"}>{geometryConfidence>=.8?<Check size={13}/>:<AlertTriangle size={13}/>} Geometry {Math.round(geometryConfidence*100)}%</span><button type="button" onClick={()=>resetCameraRef.current?.()}><Maximize size={14}/> Fit</button></div>
    <div className="spatial-model-issues" aria-label="3D issue pins">{pins.slice(0,8).map(pin=><button className={pin.issueId===selectedIssueId?"active":""} type="button" key={pin.issueId} onClick={()=>onSelectIssue(pin.issueId)}><span>{pin.code}</span><b>{pin.title}</b>{pin.issueId===selectedIssueId?<small>Selected</small>:null}</button>)}</div>
  </section>;
}
