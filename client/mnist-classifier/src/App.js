
import Canvas from './Components/Canvas'
import { Button } from '@material-ui/core';
import Grid from '@material-ui/core/Grid';
import SearchIcon from '@material-ui/icons/Search';
import './App.css'
import createMixins from '@material-ui/core/styles/createMixins';

function App() {
  let centerStyle = {display: 'flex', alignItems: 'center', justifyContent: 'center'}

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
    }
    img.setAttribute("src", canvas.toDataURL())

    // console.log(pixAlphaData)
    // console.log(canvas.toDataURL())

  }

  return (
    <div>
      <Grid container>
        <Grid style={centerStyle} item xs={12}>
          <h1>Title</h1>
        </Grid>
      </Grid>
      <Grid container>
        <Grid style={centerStyle} item xs={12}>
          <Canvas />
          <img id="image" src="" style={{display: "none"}}></img>
        </Grid>
        <Grid style={{...centerStyle, marginTop: '1rem'}} item xs={12}>
          <Button
            variant="contained"
            color="primary"
            size="small"
            startIcon={<SearchIcon />}
            onClick={classifyNum}
          >
            Classify
          </Button>
        </Grid>
      </Grid>
    </div>
  );
}

export default App;
