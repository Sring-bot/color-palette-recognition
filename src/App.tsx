import React, { useState, useRef, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Upload, Palette, Copy, Download, Save, X, Github } from 'lucide-react';
import colorNamer from 'color-namer';

interface Color {
  hex: string;
  name: string;
  percentage: number;
}

interface SavedPalette {
  id: string;
  colors: Color[];
  timestamp: number;
}

function App() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [colors, setColors] = useState<Color[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [savedPalettes, setSavedPalettes] = useState<SavedPalette[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };
  const kMeansClustering = async (pixels: number[][], k: number): Promise<number[][]> => {
  const numAttempts = 5;
  const iterations = 50;

  let bestCentroids: number[][] = [];
  let bestScore = Infinity;

  for (let attempt = 0; attempt < numAttempts; attempt++) {
    const points = tf.tensor2d(pixels);
    let centroids = points.gather(tf.randomUniform([k], 0, points.shape[0], 'int32'));

    for (let i = 0; i < iterations; i++) {
      const broadcast = tf.broadcastTo(tf.expandDims(points), [k, points.shape[0], points.shape[1]]);
      const centroidBroadcast = tf.broadcastTo(tf.expandDims(centroids, 1), [k, points.shape[0], points.shape[1]]);
      const distances = tf.sum(tf.square(tf.sub(broadcast, centroidBroadcast)), 2);
      const labels = tf.argMin(distances, 0) as tf.Tensor1D;

      const totals = tf.matMul(
        tf.transpose(tf.oneHot(labels, k)),
        points
      );

      const counts = tf.sum(tf.oneHot(labels, k), 0);
      centroids = tf.div(totals, tf.expandDims(counts, 1));
    }
    
    // Evaluate clustering: total sum of squared distances
    const broadcastFinal = tf.broadcastTo(tf.expandDims(points), [k, points.shape[0], points.shape[1]]);
    const centroidFinal = tf.broadcastTo(tf.expandDims(centroids, 1), [k, points.shape[0], points.shape[1]]);
    const finalDistances = tf.sum(tf.square(tf.sub(broadcastFinal, centroidFinal)), 2);
    const minDistances = tf.min(finalDistances, 0); // closest centroid for each point
    const scoreTensor = tf.sum(minDistances); // total intra-cluster distance
    const score = (await scoreTensor.data())[0];

    if (score < bestScore) {
      bestScore = score;
      bestCentroids = await centroids.array();
    }

    tf.dispose([points, centroids, finalDistances, scoreTensor, broadcastFinal, centroidFinal]);
  }

  return bestCentroids;
};


  const processImage = async (imageUrl: string) => {
    setIsProcessing(true);
    try {
      const img = new Image();
      img.src = imageUrl;
      await new Promise((resolve) => (img.onload = resolve));

      const tensor = tf.browser.fromPixels(img)
        .resizeNearestNeighbor([50, 50])
        .toFloat()
        .div(255.0);

      const imageData = await tf.browser.toPixels(tensor as tf.Tensor3D);
      const pixels: number[][] = [];

      for (let i = 0; i < imageData.length; i += 4) {
        pixels.push([
          imageData[i] / 255,
          imageData[i + 1] / 255,
          imageData[i + 2] / 255,
        ]);
      }

      const centroids = await kMeansClustering(pixels, 5);
      const processedColors = centroids.map(([r, g, b]) => {
        const hex = `#${Math.round(r * 255).toString(16).padStart(2, '0')}${Math.round(g * 255).toString(16).padStart(2, '0')}${Math.round(b * 255).toString(16).padStart(2, '0')}`;
        const names = colorNamer(hex);
        return {
          hex,
          name: names.ntc[0].name,
          percentage: Math.round((1 / centroids.length) * 100),
        };
      });

      setColors(processedColors);
    } catch (error) {
      console.error('Error processing image:', error);
    }
    setIsProcessing(false);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const savePalette = () => {
    if (colors.length > 0) {
      const newPalette: SavedPalette = {
        id: Date.now().toString(),
        colors,
        timestamp: Date.now(),
      };
      setSavedPalettes(prev => [...prev, newPalette]);
    }
  };

  const downloadPalette = () => {
    if (colors.length > 0) {
      const data = {
        colors,
        exportedAt: new Date().toISOString(),
      };
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `palette-${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    }
  };

  useEffect(() => {
    if (selectedImage) {
      processImage(selectedImage);
    }
  }, [selectedImage]);

  return (
    <div className="min-h-screen #ffffff py-12 px-4">
      <div className="container mx-auto max-w-4xl">
        <div className="absolute top-4 right-4">
          <a
            href="https://github.com/Sring-bot/color-palette-recognition"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-4 py-2 glass-effect2 rounded-lg hover:bg-gray-300 transition-colors"
          >
            <Github className="w-6 h-6" />
            <span className="text-sm"><b>GitHub</b></span>
          </a>
        </div>

        <div className="text-center mb-12 animate-fade-in">
          <h1 className="text-2xl font-bold mb-4 flex items-center justify-center">
            <Palette className="w-10 h-10 text mr-3" />
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 to-emerald-800">
              Color Palette Recognition
            </span>
          </h1>
          <p className="text-gray-300 text-lg">
            Extract color palettes from images using machine learning
          </p>
        </div>

        <div className="glass-effect rounded-2xl p-8 mb-8 animate-fade-in">

          <div className="flex justify-center gap-4 mb-6">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center px-4 py-2 bg-[#002d17] text-[#7fffb0] border border-[#005c32] rounded-lg hover:bg-[#004122]"
            >
              <Upload className="w-5 h-5 mr-2" />
              Upload Image
            </button>
          </div>

          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            accept="image/*"
            onChange={handleImageUpload}
          />

          <div
            className={`border-2 border-dashed border-gray-700 rounded-xl p-8 text-center transition-all duration-300 ${
              selectedImage ? 'bg-gray-800/30' : 'hover:border-emerald-700 cursor-pointer'
            }`}
            onClick={() => !selectedImage && fileInputRef.current?.click()}
          >
            {selectedImage ? (
              <div className="relative">
                <img
                  src={selectedImage}
                  alt="Uploaded"
                  className="max-h-96 mx-auto rounded-lg"
                />
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedImage(null);
                    setColors([]);
                  }}
                  className="absolute top-2 right-2 p-1 bg-gray-900/80 rounded-full hover:bg-gray-800"
                >
                  <X className="w-5 h-5 text-gray-300" />
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                <Upload className="w-12 h-12 mx-auto text-blackd" />
                <p className="text-emerald-950">Click to upload or drag and drop</p>
                
              </div>
            )}
          </div>
        </div>

        {isProcessing && (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-emerald-900 border-t-transparent mx-auto"></div>
            <p className="mt-4 text-gray-400">Processing...</p>
          </div>
        )}

        {colors.length > 0 && !isProcessing && (
          <div className="glass-effect rounded-2xl p-8 animate-fade-in">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-semibold text-black">
                Extracted Palette
              </h2>
              <div className="flex gap-3">
                <button
                  onClick={savePalette}
                  className="flex items-center px-3 py-2 bg-emerald-700/80 text-white rounded-lg hover:bg-green-900/80 transition-colors"
                >
                  <Save className="w-4 h-4 mr-2" />
                  Save
                </button>
                <button
                  onClick={downloadPalette}
                  className="flex items-center px-3 py-2 bg-rose-500/80 text-white rounded-lg hover:bg-rose-700/80 transition-colors"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Export
                </button>
              </div>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {colors.map((color, index) => (
                <div
                  key={index}
                  className="bg-rose-800/50 rounded-lg overflow-hidden transform hover:scale-105 transition-transform duration-200"
                >
                  <div
                    className="h-16 w-full"
                    style={{ backgroundColor: color.hex }}
                  />
                  <div className="p-2">
                    <div className="flex items-center justify-between mb-1">
                      <p className="font-mono text-xs text-white">
                        {color.hex.toUpperCase()}
                      </p>
                      <button
                        onClick={() => copyToClipboard(color.hex)}
                        className="p-1 hover:bg-gray-800/50 rounded"
                      >
                        <Copy className="w-3 h-3 text-white" />
                      </button>
                    </div>
                    <p className="text-xs font-medium text-white truncate" title={color.name}>
                      {color.name}
                    </p>
                    <p className="text-xs text-white">{color.percentage}%</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {savedPalettes.length > 0 && (
          <div className="mt-12 glass-effect rounded-2xl p-8">
            <h2 className="text-2xl font-semibold text-white mb-6">
              Saved Palettes
            </h2>
            <div className="space-y-4">
              {savedPalettes.map((palette) => (
                <div key={palette.id} className="bg-gray-800/30 rounded-lg p-3 hover:bg-gray-700/30 transition-colors">
                  <div className="flex gap-1 h-8">
                    {palette.colors.map((color) => (
                      <div
                        key={color.hex}
                        className="flex-1 rounded-md"
                        style={{ backgroundColor: color.hex }}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
