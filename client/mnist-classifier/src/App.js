
import Canvas from './Components/Canvas'
import { Button } from '@material-ui/core';
import Grid from '@material-ui/core/Grid';
import SearchIcon from '@material-ui/icons/Search';
import ClearIcon from '@material-ui/icons/Clear';
import './App.css'
import axios from 'axios'
import createMixins from '@material-ui/core/styles/createMixins';

function App() {
  let centerStyle = {display: 'flex', alignItems: 'center', justifyContent: 'center'}

  const clearCanvas = () => {
    let canvas = document.getElementById("canvas")
    let ctx = canvas.getContext("2d")
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  const classifyNum = () => {

    let canvas = document.getElementById("canvas")

    let img = document.getElementById("image");
    img.onload = function() {
      let canvasResized = document.createElement("canvas");
      canvasResized.width = 28;
      canvasResized.height = 28;
      let ctxResized = canvasResized.getContext("2d");
      ctxResized.drawImage(img, 0, 0, 28, 28);

      let imgd = ctxResized.getImageData(0, 0, 28, 28);
      let pix = imgd.data;
      let pixAlphaData = []
       
      // Loop over each pixel and invert the color.
      for (var i = 3, n = pix.length; i < n; i += 4) {
        pixAlphaData.push(pix[i])
      }
      console.log(pixAlphaData)

      axios.post('http://127.0.0.1:5000/api/classify', {"pixelMatrix": pixAlphaData})
        .then(response => alert(response.data.pred[0]));

    }
    img.setAttribute("src", canvas.toDataURL())
  }

  return (
    <div>
      <Grid container>
        <Grid style={centerStyle} item xs={12}>
          <h1>MNIST Classifier</h1>
        </Grid>
      </Grid>
      <Grid container>
        <Grid style={{...centerStyle, flexDirection: 'column'}} item xs={12}>
          <Canvas />
          <div style={{flexDirection: 'row', width: '280px'}}>
            <Button
              style={{ width: '50%' }}
              variant="contained"
              color="primary"
              size="small"
              startIcon={<SearchIcon />}
              onClick={classifyNum}
            >
              Classify
            </Button>
            <Button
              style={{ width: '50%' }}
              variant="contained"
              color="secondary"
              size="small"
              startIcon={<ClearIcon />}
              onClick={clearCanvas}
            >
              Clear
            </Button>
          </div>
          <img id="image" src="" style={{display: "none"}}></img>
        </Grid>
      </Grid>
    </div>
  );
}

export default App;
