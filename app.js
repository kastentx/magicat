const fs = require('fs')
const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node-gpu')

const MODEL_PATH = 'https://raw.githubusercontent.com/kastentx/tfjs-conversion/master/second-try/tensorflowjs_model.pb'
const WEIGHTS_PATH = 'https://raw.githubusercontent.com/kastentx/tfjs-conversion/master/second-try/weights_manifest.json'
const filename = process.argv[2]

const loadTFJSModel = () => 
  tf.loadModel(MODEL_PATH, WEIGHTS_PATH)
const base64_encode = file => 
  new Buffer.from(fs.readFileSync(file)).toString('base64')

console.log(`using backend: ${ tf.getBackend() }`)

try {
  async () => await loadTFJSModel()  
} catch (e) {
  console.error(`error loading model - ${e}`)
}

if (filename) {
  console.log(`submitted filename: ${ filename }`)
  console.log(`submitted image: ${ base64_encode(filename) }`)
}