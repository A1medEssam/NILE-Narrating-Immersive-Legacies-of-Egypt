from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langserve import add_routes
from Ramesses_rag_stone import onChain

app = FastAPI()

chain = onChain()

add_routes(app, chain, path='/chat')

@app.post("/process_input")
async def process_input(request: Request):
    data = await request.json()
    user_input = data.get("user_input", "")

    response = await chain.invoke({
        "input": user_input,
    })
    return JSONResponse(content={"response": response})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=9000)



